"""Unit tests for RedisManager"""

from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError

from aegra_api.core.redis_manager import RedisManager


class TestRedisManager:
    """Test RedisManager lifecycle"""

    @pytest.mark.asyncio
    async def test_initialize_creates_pool_and_pings(self) -> None:
        """Test that initialize creates connection pool and verifies connectivity"""
        manager = RedisManager()

        mock_client = AsyncMock()
        mock_pool = AsyncMock()

        with (
            patch("aegra_api.core.redis_manager.aioredis.ConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.redis_manager.aioredis.Redis", return_value=mock_client) as mock_redis_cls,
        ):
            mock_pool_cls.from_url.return_value = mock_pool

            await manager.initialize()

            mock_pool_cls.from_url.assert_called_once()
            mock_redis_cls.assert_called_once_with(connection_pool=mock_pool)
            mock_client.ping.assert_awaited_once()

        # Clean up
        manager._client = None
        manager._pool = None

    @pytest.mark.asyncio
    async def test_initialize_configures_retry_and_health_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pooled connections must survive server-side idle disconnects (#505).

        Every connection — including the one created by initialize()'s own
        PING — must carry a retry that covers ConnectionError, plus a
        health-check interval. Non-default settings prove the values come
        from RedisSettings rather than literals.
        """
        from aegra_api.settings import settings

        monkeypatch.setattr(settings.redis, "REDIS_HEALTH_CHECK_INTERVAL", 45)
        monkeypatch.setattr(settings.redis, "REDIS_RETRY_ATTEMPTS", 5)
        manager = RedisManager()
        mock_client = AsyncMock()

        with (
            patch("aegra_api.core.redis_manager.aioredis.ConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.redis_manager.aioredis.Redis", return_value=mock_client),
        ):
            await manager.initialize()

            kwargs = mock_pool_cls.from_url.call_args.kwargs
            assert kwargs["health_check_interval"] == 45
            retry = kwargs["retry"]
            assert isinstance(retry, Retry)
            assert retry.get_retries() == 5
            assert isinstance(retry._backoff, ExponentialBackoff)
            assert any(issubclass(RedisConnectionError, e) for e in retry._supported_errors)
            assert RedisConnectionError in kwargs["retry_on_error"]

        manager._client = None
        manager._pool = None

    @pytest.mark.asyncio
    async def test_real_pool_connections_carry_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end through the real ConnectionPool: connections built from
        the pool's kwargs must have retries (asyncio default is 0)."""
        from aegra_api.settings import settings

        monkeypatch.setattr(settings.redis, "REDIS_HEALTH_CHECK_INTERVAL", 45)
        monkeypatch.setattr(settings.redis, "REDIS_RETRY_ATTEMPTS", 5)
        manager = RedisManager()
        mock_client = AsyncMock()
        real_from_url = aioredis.ConnectionPool.from_url
        captured: dict[str, aioredis.ConnectionPool] = {}

        def capture_from_url(url: str, **kwargs: object) -> aioredis.ConnectionPool:
            captured["pool"] = real_from_url("redis://localhost:1/0", **kwargs)
            return captured["pool"]

        with (
            patch(
                "aegra_api.core.redis_manager.aioredis.ConnectionPool.from_url",
                side_effect=capture_from_url,
            ),
            patch("aegra_api.core.redis_manager.aioredis.Redis", return_value=mock_client),
        ):
            await manager.initialize()

        conn = captured["pool"].make_connection()
        assert conn.retry.get_retries() == 5
        assert conn.health_check_interval == 45

        manager._client = None
        manager._pool = None

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self) -> None:
        """Test that calling initialize twice doesn't create a second pool"""
        manager = RedisManager()
        manager._client = AsyncMock()  # Simulate already initialized

        with patch("aegra_api.core.redis_manager.aioredis.ConnectionPool") as mock_pool_cls:
            await manager.initialize()

            mock_pool_cls.from_url.assert_not_called()

        # Clean up
        manager._client = None

    @pytest.mark.asyncio
    async def test_close_cleans_up(self) -> None:
        """Test that close disposes of client and pool"""
        manager = RedisManager()
        mock_client = AsyncMock()
        mock_pool = AsyncMock()
        manager._client = mock_client
        manager._pool = mock_pool

        await manager.close()

        mock_client.aclose.assert_awaited_once()
        mock_pool.disconnect.assert_awaited_once()
        assert manager._client is None
        assert manager._pool is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self) -> None:
        """Test that close is safe when not initialized"""
        manager = RedisManager()

        # Should not raise
        await manager.close()

    def test_get_client_returns_client(self) -> None:
        """Test get_client returns the initialized client"""
        manager = RedisManager()
        mock_client = AsyncMock()
        manager._client = mock_client

        result = manager.get_client()

        assert result is mock_client

        # Clean up
        manager._client = None

    def test_get_client_raises_when_not_initialized(self) -> None:
        """Test get_client raises RuntimeError when not initialized"""
        manager = RedisManager()

        with pytest.raises(RuntimeError, match="Redis not initialized"):
            manager.get_client()
