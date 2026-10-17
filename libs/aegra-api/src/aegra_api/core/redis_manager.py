"""Redis connection manager for the event broker."""

from typing import Any
from urllib.parse import urlparse

import redis.asyncio as aioredis
import structlog
from redis.asyncio.retry import Retry
from redis.asyncio.sentinel import Sentinel
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from aegra_api.settings import SentinelConfig, settings

logger = structlog.get_logger(__name__)


class RedisManager:
    """Manages Redis connection pool lifecycle.

    Follows the same pattern as DatabaseManager: a global singleton
    initialized during app lifespan and closed on shutdown.

    Connects one of two ways, chosen by REDIS_URL's scheme. A ``redis://``
    URL dials that endpoint directly. A ``redis+sentinel://`` URL (or
    ``rediss+sentinel://`` for TLS) goes through Sentinel instead, which
    resolves the current master at connect time and re-resolves it after a
    failover. Both paths hand back the same ``redis.asyncio.Redis``, so
    nothing downstream of ``get_client()`` knows the difference.
    """

    def __init__(self) -> None:
        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None
        self._sentinel: Sentinel | None = None

    @staticmethod
    def _connection_kwargs() -> dict[str, Any]:
        """Connection settings shared by the direct and Sentinel paths.

        Built fresh per call so each pool gets its own Retry instance.
        Directly constructed pools default to zero retries; bounded retries
        survive idle disconnects and failovers. Lease acquisition deduplicates
        RPUSH replays (#505).
        """
        return {
            "decode_responses": True,
            "health_check_interval": settings.redis.REDIS_HEALTH_CHECK_INTERVAL,
            "retry": Retry(
                ExponentialBackoff(cap=1.0, base=0.05),
                settings.redis.REDIS_RETRY_ATTEMPTS,
            ),
            "retry_on_error": [RedisConnectionError, RedisTimeoutError],
        }

    def _connect_direct(self) -> None:
        """Dial REDIS_URL directly."""
        self._pool = aioredis.ConnectionPool.from_url(
            settings.redis.REDIS_URL,
            max_connections=settings.redis.REDIS_MAX_CONNECTIONS,
            **self._connection_kwargs(),
        )
        self._client = aioredis.Redis(connection_pool=self._pool)

    def _connect_sentinel(self, config: SentinelConfig) -> None:
        """Resolve the master through Sentinel.

        The sentinels and the data nodes are separate servers with separate
        credentials, so they get separate connection kwargs. ``master_for``
        returns a client whose pool re-queries the sentinels whenever it needs
        a connection, which is what makes it follow a failover.

        On the TLS scheme both hops are wrapped: ``ssl=True`` makes
        ``Redis`` pick ``SSLConnection`` for the sentinels, and
        ``SentinelConnectionPool`` pick ``SentinelManagedSSLConnection`` for
        the master.
        """
        sentinel_kwargs = self._connection_kwargs()
        if config.sentinel_username is not None:
            sentinel_kwargs["username"] = config.sentinel_username
        if config.sentinel_password is not None:
            sentinel_kwargs["password"] = config.sentinel_password

        master_kwargs = self._connection_kwargs()
        master_kwargs["db"] = config.db
        if config.username is not None:
            master_kwargs["username"] = config.username
        if config.password is not None:
            master_kwargs["password"] = config.password

        if config.ssl:
            for kwargs in (sentinel_kwargs, master_kwargs):
                kwargs["ssl"] = True
                kwargs.update(config.ssl_options)

        self._sentinel = Sentinel(
            list(config.hosts),
            sentinel_kwargs=sentinel_kwargs,
            **master_kwargs,
        )
        self._client = self._sentinel.master_for(
            config.master_name,
            max_connections=settings.redis.REDIS_MAX_CONNECTIONS,
        )
        # Tracked so close() tears the master pool down the same way it does
        # for the direct path.
        self._pool = self._client.connection_pool

    async def initialize(self) -> None:
        """Create the connection pool and verify connectivity."""
        if self._client is not None:
            return

        sentinel_config = settings.redis.sentinel
        if sentinel_config is not None:
            self._connect_sentinel(sentinel_config)
        else:
            self._connect_direct()

        await self._client.ping()  # type: ignore[invalid-await]  # redis.asyncio stubs

        # Log endpoints only, never the URL itself, which may carry credentials
        if sentinel_config is not None:
            logger.info(
                "Redis broker initialized via Sentinel",
                sentinels=[f"{host}:{port}" for host, port in sentinel_config.hosts],
                master_name=sentinel_config.master_name,
                tls=sentinel_config.ssl,
            )
        else:
            parsed = urlparse(settings.redis.REDIS_URL)
            logger.info("Redis broker initialized", host=parsed.hostname, port=parsed.port)

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        if self._sentinel:
            # The sentinel clients hold pools of their own, separate from the
            # master pool above.
            for sentinel_client in self._sentinel.sentinels:
                await sentinel_client.aclose()
            self._sentinel = None
        logger.info("Redis broker connections closed")

    def get_client(self) -> aioredis.Redis:
        """Return the shared async Redis client."""
        if self._client is None:
            raise RuntimeError("Redis not initialized. Set REDIS_BROKER_ENABLED=true and ensure Redis is running.")
        return self._client


# Global Redis manager instance
redis_manager = RedisManager()
