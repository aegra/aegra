"""Unit tests for thread TTL config resolution and the expiry sweep."""

import json

import pytest
from pydantic import ValidationError

from aegra_api.services.thread_ttl import (
    ThreadTTLConfig,
    get_thread_ttl_config,
)
from aegra_api.settings import settings


@pytest.fixture(autouse=True)
def _clear_ttl_config_cache() -> None:
    get_thread_ttl_config.cache_clear()
    yield
    get_thread_ttl_config.cache_clear()


@pytest.fixture(autouse=True)
def _no_env_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", None)


class TestResolveConfig:
    """Tests for get_thread_ttl_config source precedence and validation."""

    def test_returns_none_when_no_source_configured(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        assert get_thread_ttl_config() is None

    def test_loads_from_aegra_json_checkpointer_ttl(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "checkpointer": {"ttl": {"strategy": "keep_latest", "default_ttl": 1440}},
                }
            )
        )

        config = get_thread_ttl_config()

        assert config is not None
        assert config.strategy == "keep_latest"
        assert config.default_ttl == 1440
        assert config.sweep_interval_minutes == 5
        assert config.sweep_limit == 1000

    def test_env_bare_number_sets_default_ttl(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "43200")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 43200
        assert config.strategy == "delete"

    def test_env_json_object_sets_all_fields(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            settings.thread_ttl,
            "AEGRA_THREAD_TTL",
            json.dumps(
                {
                    "strategy": "keep_latest",
                    "default_ttl": 60,
                    "sweep_interval_minutes": 1,
                    "sweep_limit": 50,
                }
            ),
        )

        config = get_thread_ttl_config()

        assert config is not None
        assert config.strategy == "keep_latest"
        assert config.default_ttl == 60
        assert config.sweep_interval_minutes == 1
        assert config.sweep_limit == 50

    def test_env_replaces_aegra_json_block_entirely(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whole-source override: json keys do not leak under an env config."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "checkpointer": {"ttl": {"strategy": "keep_latest", "sweep_limit": 7}},
                }
            )
        )
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "120")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 120
        assert config.strategy == "delete"
        assert config.sweep_limit == 1000

    def test_invalid_env_json_raises(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "{not json")

        with pytest.raises(json.JSONDecodeError):
            get_thread_ttl_config()

    def test_invalid_strategy_raises(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", json.dumps({"strategy": "purge"}))

        with pytest.raises(ValidationError):
            get_thread_ttl_config()

    def test_non_positive_default_ttl_raises(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "0")

        with pytest.raises(ValidationError):
            get_thread_ttl_config()

    def test_blank_env_falls_back_to_json(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "checkpointer": {"ttl": {"default_ttl": 15}},
                }
            )
        )
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "   ")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 15


class TestThreadTTLConfigModel:
    """Bounds validation on the config model itself."""

    def test_defaults(self) -> None:
        config = ThreadTTLConfig()

        assert config.strategy == "delete"
        assert config.default_ttl == 43200
        assert config.sweep_interval_minutes == 5
        assert config.sweep_limit == 1000

    @pytest.mark.parametrize(
        "field",
        ["default_ttl", "sweep_interval_minutes", "sweep_limit"],
    )
    def test_rejects_non_positive_values(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ThreadTTLConfig.model_validate({field: 0})
