"""Unit tests for loading the cron firing-user resolver named by ``auth.cron_user``."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import FunctionType

import pytest

from aegra_api.core.auth_middleware import LangGraphAuthBackend
from aegra_api.core.cron_user import load_cron_user_resolver
from aegra_api.settings import settings

AUTH_FILE = """
from langgraph_sdk import Auth

auth = Auth()


@auth.authenticate
async def authenticate(headers: dict) -> dict:
    return {"identity": "alice"}


async def cron_user(identity: str) -> dict | None:
    return {"identity": identity, "tenant_id": "acme"}


def sync_cron_user(identity: str) -> dict:
    return {"identity": identity}
"""


@pytest.fixture(autouse=True)
def restore_sys_modules() -> Iterator[None]:
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


def _write_config(directory: Path, auth: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "aegra.json"
    config_path.write_text(json.dumps({"graphs": {}, "auth": auth}))
    (directory / "deploy_auth.py").write_text(AUTH_FILE)
    return config_path


def test_returns_none_without_a_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert load_cron_user_resolver() is None


def test_returns_none_when_auth_names_no_cron_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"path": "./deploy_auth.py:auth"})
    monkeypatch.chdir(tmp_path)

    assert load_cron_user_resolver() is None


@pytest.mark.asyncio
async def test_loads_a_file_relative_to_the_config_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path / "deploy", {"cron_user": "./deploy_auth.py:cron_user"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings.app, "AEGRA_CONFIG", str(config_path))

    resolver = load_cron_user_resolver()

    assert resolver is not None
    assert await resolver("alice") == {"identity": "alice", "tenant_id": "acme"}


@pytest.mark.asyncio
async def test_loads_from_an_installed_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"cron_user": "deploy_auth:cron_user"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    resolver = load_cron_user_resolver()

    assert resolver is not None
    assert await resolver("bob") == {"identity": "bob", "tenant_id": "acme"}


def test_rejects_a_path_without_a_variable_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"cron_user": "./deploy_auth.py"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="missing ':'"):
        load_cron_user_resolver()


def test_rejects_a_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"cron_user": "./missing.py:cron_user"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing.py"):
        load_cron_user_resolver()


def test_rejects_a_file_python_has_no_loader_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"cron_user": "./deploy_auth.txt:cron_user"})
    (tmp_path / "deploy_auth.txt").write_text(AUTH_FILE)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ImportError, match="Could not load module"):
        load_cron_user_resolver()


def test_rejects_a_missing_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"cron_user": "./deploy_auth.py:no_such_resolver"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AttributeError, match="no_such_resolver"):
        load_cron_user_resolver()


def test_rejects_a_function_that_is_not_async(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"cron_user": "./deploy_auth.py:sync_cron_user"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TypeError, match="async function"):
        load_cron_user_resolver()


def test_shares_the_module_the_auth_handler_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"path": "./deploy_auth.py:auth", "cron_user": "./deploy_auth.py:cron_user"})
    monkeypatch.chdir(tmp_path)

    backend = LangGraphAuthBackend()
    resolver = load_cron_user_resolver()

    assert isinstance(resolver, FunctionType)
    assert resolver.__globals__["auth"] is backend.auth_instance


@pytest.mark.asyncio
async def test_imports_a_second_file_with_the_same_stem_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, {"path": "./deploy_auth.py:auth", "cron_user": "./crons/deploy_auth.py:cron_user"})
    (tmp_path / "crons").mkdir()
    (tmp_path / "crons" / "deploy_auth.py").write_text(AUTH_FILE.replace('"acme"', '"globex"'))
    monkeypatch.chdir(tmp_path)

    LangGraphAuthBackend()
    resolver = load_cron_user_resolver()

    assert resolver is not None
    assert await resolver("alice") == {"identity": "alice", "tenant_id": "globex"}


def test_the_auth_handler_shares_the_module_the_resolver_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, {"path": "./deploy_auth.py:auth", "cron_user": "./deploy_auth.py:cron_user"})
    monkeypatch.chdir(tmp_path)

    resolver = load_cron_user_resolver()
    backend = LangGraphAuthBackend()

    assert isinstance(resolver, FunctionType)
    assert backend.auth_instance is resolver.__globals__["auth"]


@pytest.mark.parametrize(
    "auth",
    [
        {"path": "deploy_auth:auth", "cron_user": "./deploy_auth.py:cron_user"},
        {"path": "./deploy_auth.py:auth", "cron_user": "deploy_auth:cron_user"},
    ],
    ids=["path-as-module", "cron-user-as-module"],
)
@pytest.mark.parametrize("resolver_first", [True, False], ids=["resolver-first", "handler-first"])
def test_a_file_named_as_a_module_and_as_a_path_is_imported_once_as_the_module(
    auth: dict[str, str], resolver_first: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, auth)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    if resolver_first:
        resolver = load_cron_user_resolver()
        backend = LangGraphAuthBackend()
    else:
        backend = LangGraphAuthBackend()
        resolver = load_cron_user_resolver()

    assert isinstance(resolver, FunctionType)
    assert resolver.__globals__["auth"] is backend.auth_instance
    assert sys.modules["deploy_auth"].auth is backend.auth_instance


@pytest.mark.parametrize("cron_user_form", ["file", "module"])
def test_a_file_spelled_with_dot_dot_is_the_file_it_resolves_to(
    cron_user_form: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detour = tmp_path / "detour"
    detour.mkdir()
    cron_user = {"file": f"{detour}/../deploy_auth.py:cron_user", "module": "deploy_auth:cron_user"}[cron_user_form]
    _write_config(tmp_path, {"path": "./deploy_auth.py:auth", "cron_user": cron_user})
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(detour / ".."))

    backend = LangGraphAuthBackend()
    resolver = load_cron_user_resolver()

    assert isinstance(resolver, FunctionType)
    assert resolver.__globals__["auth"] is backend.auth_instance


@pytest.mark.asyncio
async def test_imports_a_file_with_the_stem_of_a_module_named_elsewhere_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, {"path": "deploy_auth:auth", "cron_user": "./crons/deploy_auth.py:cron_user"})
    (tmp_path / "crons").mkdir()
    (tmp_path / "crons" / "deploy_auth.py").write_text(AUTH_FILE.replace('"acme"', '"globex"'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    backend = LangGraphAuthBackend()
    resolver = load_cron_user_resolver()

    assert resolver is not None
    assert await resolver("alice") == {"identity": "alice", "tenant_id": "globex"}
    assert backend.auth_instance is sys.modules["deploy_auth"].auth


@pytest.mark.parametrize(
    ("cron_user", "error"),
    [
        ("no_such_package.deploy_auth:cron_user", ModuleNotFoundError),
        ("no_such_module:cron_user", ModuleNotFoundError),
        ("deploy_auth", ValueError),
    ],
)
def test_a_cron_user_path_that_does_not_load_leaves_the_auth_handler_loading(
    cron_user: str, error: type[Exception], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, {"path": "./deploy_auth.py:auth", "cron_user": cron_user})
    monkeypatch.chdir(tmp_path)

    backend = LangGraphAuthBackend()

    assert backend.auth_instance is not None
    with pytest.raises(error):
        load_cron_user_resolver()
