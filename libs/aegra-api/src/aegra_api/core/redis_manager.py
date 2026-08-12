"""Redis connection manager for the event broker."""

from urllib.parse import urlparse

import redis.asyncio as aioredis
import structlog
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

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

        # Directly constructed pools default to zero retries; bounded retries survive
        # idle disconnects/failovers. Lease acquisition deduplicates RPUSH replays (#505).
        self._pool = aioredis.ConnectionPool.from_url(
            settings.redis.REDIS_URL,
            max_connections=settings.redis.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
            health_check_interval=settings.redis.REDIS_HEALTH_CHECK_INTERVAL,
            retry=Retry(
                ExponentialBackoff(cap=1.0, base=0.05),
                settings.redis.REDIS_RETRY_ATTEMPTS,
            ),
            retry_on_error=[RedisConnectionError, RedisTimeoutError],
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


# Global Redis manager instance
redis_manager = RedisManager()
