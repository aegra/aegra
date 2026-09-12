"""Unit tests for RedisManager"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis
from redis.asyncio.connection import SSLConnection
from redis.asyncio.retry import Retry
from redis.asyncio.sentinel import (
    Sentinel,
    SentinelManagedConnection,
    SentinelManagedSSLConnection,
)
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError

from aegra_api.core import redis_manager as redis_manager_module
from aegra_api.core.redis_manager import RedisManager
from aegra_api.settings import RedisSettings


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
    async def test_initialize_configures_retry_and_health_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pooled connections must survive server-side idle disconnects (#505).

        Non-default settings prove the values come from RedisSettings rather
        than literals. Patched through the settings object redis_manager
        holds, since a sibling test reloads aegra_api.settings.
        """
        monkeypatch.setattr(redis_manager_module.settings.redis, "REDIS_HEALTH_CHECK_INTERVAL", 45)
        monkeypatch.setattr(redis_manager_module.settings.redis, "REDIS_RETRY_ATTEMPTS", 5)
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
    async def test_real_pool_connections_carry_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end through the real ConnectionPool: connections built from
        the pool's kwargs must have retries (asyncio default is 0)."""
        monkeypatch.setattr(redis_manager_module.settings.redis, "REDIS_HEALTH_CHECK_INTERVAL", 45)
        monkeypatch.setattr(redis_manager_module.settings.redis, "REDIS_RETRY_ATTEMPTS", 5)
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


class TestRedisManagerSentinel:
    """Test the optional redis+sentinel:// connection path."""

    @staticmethod
    def _use_sentinel_url(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
        """Point the settings object redis_manager holds at a sentinel URL."""
        parsed = RedisSettings(_env_file=None, REDIS_URL=url)
        monkeypatch.setattr(redis_manager_module.settings, "redis", parsed)

    @pytest.mark.asyncio
    async def test_direct_path_is_untouched_by_default(self) -> None:
        """The default URL still goes through ConnectionPool.from_url."""
        manager = RedisManager()
        mock_client = AsyncMock()

        with (
            patch("aegra_api.core.redis_manager.aioredis.ConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.redis_manager.aioredis.Redis", return_value=mock_client),
            patch("aegra_api.core.redis_manager.Sentinel") as mock_sentinel_cls,
        ):
            await manager.initialize()

            mock_pool_cls.from_url.assert_called_once()
            mock_sentinel_cls.assert_not_called()
            assert manager._sentinel is None

        manager._client = None
        manager._pool = None

    @pytest.mark.asyncio
    async def test_sentinel_url_resolves_master(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sentinel URL builds a Sentinel and asks it for the master."""
        self._use_sentinel_url(monkeypatch, "redis+sentinel://a.example:26379,b.example:26380/mymaster/2")
        manager = RedisManager()
        mock_client = AsyncMock()

        with (
            patch("aegra_api.core.redis_manager.aioredis.ConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.redis_manager.Sentinel") as mock_sentinel_cls,
        ):
            mock_sentinel_cls.return_value.master_for.return_value = mock_client

            await manager.initialize()

            # Never dials an endpoint directly on this path
            mock_pool_cls.from_url.assert_not_called()

            hosts = mock_sentinel_cls.call_args.args[0]
            assert hosts == [("a.example", 26379), ("b.example", 26380)]
            assert mock_sentinel_cls.call_args.kwargs["db"] == 2

            mock_sentinel_cls.return_value.master_for.assert_called_once()
            assert mock_sentinel_cls.return_value.master_for.call_args.args[0] == "mymaster"
            mock_client.ping.assert_awaited_once()
            # close() relies on the master pool being tracked like the direct one
            assert manager._pool is mock_client.connection_pool

        manager._client = None
        manager._pool = None
        manager._sentinel = None

    @pytest.mark.asyncio
    async def test_credentials_are_routed_to_the_right_servers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Data-node auth goes to the master, sentinel auth to the sentinels."""
        self._use_sentinel_url(
            monkeypatch,
            "redis+sentinel://dbuser:dbpass@a.example:26379/mymaster?sentinel_username=watcher&sentinel_password=spass",
        )
        manager = RedisManager()

        with patch("aegra_api.core.redis_manager.Sentinel") as mock_sentinel_cls:
            mock_sentinel_cls.return_value.master_for.return_value = AsyncMock()

            await manager.initialize()

            kwargs = mock_sentinel_cls.call_args.kwargs
            assert kwargs["username"] == "dbuser"
            assert kwargs["password"] == "dbpass"
            assert kwargs["sentinel_kwargs"]["username"] == "watcher"
            assert kwargs["sentinel_kwargs"]["password"] == "spass"
            # The data-node credential must not leak onto the sentinel connections
            assert kwargs["sentinel_kwargs"]["password"] != "dbpass"

        manager._client = None
        manager._pool = None
        manager._sentinel = None

    @pytest.mark.asyncio
    async def test_sentinel_path_carries_retry_and_health_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both the master and the sentinel connections get the resilience settings.

        A failover is precisely when a dead connection is retried, so this path
        needs them at least as much as the direct one.
        """
        self._use_sentinel_url(monkeypatch, "redis+sentinel://a.example:26379/mymaster")
        monkeypatch.setattr(redis_manager_module.settings.redis, "REDIS_HEALTH_CHECK_INTERVAL", 45)
        monkeypatch.setattr(redis_manager_module.settings.redis, "REDIS_RETRY_ATTEMPTS", 5)
        manager = RedisManager()

        with patch("aegra_api.core.redis_manager.Sentinel") as mock_sentinel_cls:
            mock_sentinel_cls.return_value.master_for.return_value = AsyncMock()

            await manager.initialize()

            kwargs = mock_sentinel_cls.call_args.kwargs
            for scope in (kwargs, kwargs["sentinel_kwargs"]):
                assert scope["health_check_interval"] == 45
                assert isinstance(scope["retry"], Retry)
                assert scope["retry"].get_retries() == 5
                assert RedisConnectionError in scope["retry_on_error"]
            # Separate Retry instances, not one shared between the two pools
            assert kwargs["retry"] is not kwargs["sentinel_kwargs"]["retry"]

        manager._client = None
        manager._pool = None
        manager._sentinel = None

    @pytest.mark.asyncio
    async def test_real_sentinel_object_accepts_our_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Construct a genuine Sentinel, not a mock, so a bad kwarg fails here.

        No server is contacted: building the object and its master pool is
        enough to prove redis-py accepts the arguments we pass.
        """
        self._use_sentinel_url(monkeypatch, "redis+sentinel://a.example:26379,b.example:26380/mymaster/1")
        manager = RedisManager()
        config = redis_manager_module.settings.redis.sentinel
        assert config is not None

        manager._connect_sentinel(config)

        assert isinstance(manager._sentinel, Sentinel)
        assert [(s.connection_pool.connection_kwargs["host"]) for s in manager._sentinel.sentinels] == [
            "a.example",
            "b.example",
        ]
        assert manager._client is not None
        assert manager._pool is manager._client.connection_pool
        assert manager._pool.connection_kwargs["db"] == 1

        manager._client = None
        manager._pool = None
        manager._sentinel = None

    @pytest.mark.asyncio
    async def test_close_disposes_sentinel_clients(self) -> None:
        """The sentinel connections are pooled separately from the master's."""
        manager = RedisManager()
        manager._client = AsyncMock()
        manager._pool = AsyncMock()
        sentinel_a, sentinel_b = AsyncMock(), AsyncMock()
        manager._sentinel = MagicMock(sentinels=[sentinel_a, sentinel_b])

        await manager.close()

        sentinel_a.aclose.assert_awaited_once()
        sentinel_b.aclose.assert_awaited_once()
        assert manager._sentinel is None
        assert manager._client is None
        assert manager._pool is None


class TestRedisManagerSentinelTLS:
    """Test that rediss+sentinel:// wraps both hops in TLS.

    These build genuine redis-py objects rather than mocks: the whole point is
    that the connection classes and ssl options come out right, and a mock
    would assert nothing about that.
    """

    @staticmethod
    def _connect(monkeypatch: pytest.MonkeyPatch, url: str) -> RedisManager:
        monkeypatch.setattr(
            redis_manager_module.settings,
            "redis",
            RedisSettings(_env_file=None, REDIS_URL=url),
        )
        manager = RedisManager()
        config = redis_manager_module.settings.redis.sentinel
        assert config is not None
        manager._connect_sentinel(config)
        return manager

    def test_plaintext_sentinel_uses_no_tls_connection_classes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """redis+sentinel:// must not quietly acquire TLS."""
        manager = self._connect(monkeypatch, "redis+sentinel://a.example:26379/mymaster")

        assert manager._sentinel is not None
        assert manager._pool is not None
        assert manager._pool.connection_class is SentinelManagedConnection
        assert manager._sentinel.sentinels[0].connection_pool.connection_class is not SSLConnection

        manager._client = None
        manager._pool = None
        manager._sentinel = None

    def test_tls_applies_to_sentinels_and_master(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both hops get TLS: the sentinels and the master they point at.

        Wrapping only the master would leave the discovery traffic -- and the
        sentinel AUTH that goes with it -- in the clear.
        """
        manager = self._connect(
            monkeypatch,
            "rediss+sentinel://a.example:26379,b.example:26379/mymaster/1?ssl_ca_certs=/certs/ca.pem",
        )

        assert manager._sentinel is not None
        assert manager._pool is not None
        for sentinel_client in manager._sentinel.sentinels:
            assert sentinel_client.connection_pool.connection_class is SSLConnection
            assert sentinel_client.connection_pool.connection_kwargs["ssl_ca_certs"] == "/certs/ca.pem"
        assert manager._pool.connection_class is SentinelManagedSSLConnection
        assert manager._pool.connection_kwargs["ssl_ca_certs"] == "/certs/ca.pem"

        manager._client = None
        manager._pool = None
        manager._sentinel = None

    def test_tls_options_reach_a_real_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Build an actual connection object off the master pool.

        This is the assertion that would catch redis-py renaming or dropping
        one of the ssl_* kwargs we forward.
        """
        manager = self._connect(
            monkeypatch,
            "rediss+sentinel://a.example:26379/mymaster?ssl_ca_certs=/certs/ca.pem&ssl_check_hostname=false",
        )

        assert manager._pool is not None
        connection = manager._pool.make_connection()

        assert isinstance(connection, SentinelManagedSSLConnection)
        assert connection.ca_certs == "/certs/ca.pem"
        assert connection.check_hostname is False

        manager._client = None
        manager._pool = None
        manager._sentinel = None

    def test_hostname_verification_on_by_default_on_a_real_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ssl_check_hostname in the URL, the connection still verifies.

        redis-py's async SSLConnection defaulted check_hostname to False before
        6.0, so inheriting the default would silently weaken this on redis 5.x.
        """
        manager = self._connect(monkeypatch, "rediss+sentinel://a.example:26379/mymaster?ssl_ca_certs=/certs/ca.pem")

        assert manager._pool is not None
        connection = manager._pool.make_connection()

        assert isinstance(connection, SentinelManagedSSLConnection)
        assert connection.check_hostname is True

        manager._client = None
        manager._pool = None
        manager._sentinel = None
