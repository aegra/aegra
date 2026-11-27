"""A config that names auth or a config file it cannot use must stop the boot, not drop auth."""

import json
from pathlib import Path
from typing import Any

import pytest

from aegra_api import config as config_module
from aegra_api.config import ConfigError, _resolve_config_path, load_auth_config


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module.settings.app, "AEGRA_CONFIG", None)
    return tmp_path


def _write_config(project: Path, data: Any, name: str = "aegra.json") -> None:
    (project / name).write_text(json.dumps(data))


class TestResolveConfigPath:
    def test_named_config_that_is_missing_refuses_instead_of_using_another_file(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_config(project, {"graphs": {}})
        monkeypatch.setattr(config_module.settings.app, "AEGRA_CONFIG", "prod.json")

        with pytest.raises(ConfigError, match="prod.json"):
            _resolve_config_path()

    def test_named_config_that_exists_wins_over_discovery(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_config(project, {"graphs": {}})
        _write_config(project, {"graphs": {}}, name="prod.json")
        monkeypatch.setattr(config_module.settings.app, "AEGRA_CONFIG", "prod.json")

        assert _resolve_config_path() == Path("prod.json")

    def test_default_name_still_falls_back_to_langgraph_json(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_config(project, {"graphs": {}}, name="langgraph.json")
        monkeypatch.setattr(config_module.settings.app, "AEGRA_CONFIG", "aegra.json")

        assert _resolve_config_path() == Path("langgraph.json")

    @pytest.mark.parametrize("value", [None, ""])
    def test_unset_env_uses_discovery(self, project: Path, monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
        _write_config(project, {"graphs": {}})
        monkeypatch.setattr(config_module.settings.app, "AEGRA_CONFIG", value)

        assert _resolve_config_path() == Path("aegra.json")


class TestLoadAuthConfig:
    def test_returns_auth_section_with_path(self, project: Path) -> None:
        _write_config(project, {"auth": {"path": "./auth.py:auth", "disable_studio_auth": True}})

        assert load_auth_config() == {"path": "./auth.py:auth", "disable_studio_auth": True}

    @pytest.mark.parametrize("auth", [None, {}])
    def test_absent_or_empty_auth_means_no_auth(self, project: Path, auth: dict[str, Any] | None) -> None:
        _write_config(project, {"graphs": {}, "auth": auth})

        assert load_auth_config() is None

    def test_missing_auth_key_means_no_auth(self, project: Path) -> None:
        _write_config(project, {"graphs": {}})

        assert load_auth_config() is None

    @pytest.mark.parametrize("auth", ["./auth.py:auth", ["./auth.py:auth"], True, 1, ""])
    def test_auth_that_is_not_an_object_refuses(self, project: Path, auth: Any) -> None:
        _write_config(project, {"auth": auth})

        with pytest.raises(ConfigError, match="'auth' must be an object"):
            load_auth_config()

    @pytest.mark.parametrize("key", ["paht", "Path", "auth_path"])
    def test_unknown_auth_key_refuses(self, project: Path, key: str) -> None:
        _write_config(project, {"auth": {key: "./auth.py:auth"}})

        with pytest.raises(ConfigError, match="Unknown key"):
            load_auth_config()

    @pytest.mark.parametrize("path", [None, 1, ["./auth.py:auth"]])
    def test_non_string_path_refuses(self, project: Path, path: Any) -> None:
        _write_config(project, {"auth": {"path": path}})

        with pytest.raises(ConfigError, match="must be a string"):
            load_auth_config()

    def test_langgraph_only_keys_are_accepted(self, project: Path) -> None:
        auth = {"path": "./auth.py:auth", "openapi": {"securitySchemes": {}}, "cache": {"ttl_seconds": 60}}
        _write_config(project, {"auth": auth})

        assert load_auth_config() == auth
