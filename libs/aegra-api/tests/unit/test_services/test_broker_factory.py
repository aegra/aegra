"""Unit tests for broker factory selection"""

from pathlib import Path
from unittest.mock import patch

from aegra_api.services import redis_broker
from aegra_api.services.base_broker import REPLAY_RETENTION_SECONDS
from aegra_api.services.broker import BrokerManager, _create_broker_manager
from aegra_api.services.redis_broker import RedisBrokerManager


class TestBrokerFactory:
    """Test _create_broker_manager factory function"""

    def test_returns_in_memory_broker_when_redis_disabled(self) -> None:
        with patch("aegra_api.services.broker.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = False

            manager = _create_broker_manager()

            assert isinstance(manager, BrokerManager)

    def test_returns_redis_broker_when_redis_enabled(self) -> None:
        with patch("aegra_api.services.broker.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = True

            manager = _create_broker_manager()

            assert isinstance(manager, RedisBrokerManager)


class TestReplayRetentionIsShared:
    """Dev and prod drifted to 3600s vs 600s once; pin them to one constant."""

    def test_redis_backend_uses_the_shared_retention(self) -> None:
        assert redis_broker._REPLAY_TTL_SECONDS == REPLAY_RETENTION_SECONDS

    def test_documented_window_matches_the_constant(self) -> None:
        doc = Path(__file__).parents[5] / "docs" / "guides" / "streaming.mdx"
        minutes = REPLAY_RETENTION_SECONDS // 60
        assert f"retained for {minutes} minutes" in doc.read_text(encoding="utf-8")
