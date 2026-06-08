"""Redis connection manager for the event broker."""

from urllib.parse import urlparse

import redis.asyncio as aioredis
import structlog
from redis import RedisError

from aegra_api.settings import settings

logger = structlog.get_logger(__name__)


class RedisManager:
    """Manages Redis connection pool lifecycle.

    Follows the same pattern as DatabaseManager: a global singleton
    initialized during app lifespan and closed on shutdown.
    """

    def __init__(self) -> None:
        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    async def initialize(self) -> None:
        """Create connection pool and verify connectivity."""
        if self._client is not None:
            return

        self._pool = aioredis.ConnectionPool.from_url(
            settings.redis.REDIS_URL,
            max_connections=settings.redis.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)

        await self._client.ping()  # type: ignore[invalid-await]  # redis.asyncio stubs
        # Log only host info, not full URL which may contain credentials
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
        logger.info("Redis broker connections closed")

    def get_client(self) -> aioredis.Redis:
        """Return the shared async Redis client."""
        if self._client is None:
            raise RuntimeError("Redis not initialized. Set REDIS_BROKER_ENABLED=true and ensure Redis is running.")
        return self._client

    async def is_reachable(self) -> bool:
        """Probe Redis with a PING. Returns ``True`` on success, ``False`` on
        any failure (uninitialized, connection error, timeout).

        Used at startup to seed observability gauges (``aegra_redis_up``)
        so dashboards reflect actual reachability instead of an
        optimistic default.
        """
        if self._client is None:
            return False
        try:
            await self._client.ping()  # type: ignore[invalid-await]
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("Redis PING failed", error=str(exc))
            return False
        else:
            return True


# Global Redis manager instance
redis_manager = RedisManager()
