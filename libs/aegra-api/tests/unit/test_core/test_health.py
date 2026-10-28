"""Regression tests for persistence health and readiness probes."""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegra_api.core import health


class _Connection:
    async def execute(self, _statement: Any) -> None:
        return None


class _Engine:
    def begin(self) -> "_Engine":
        return self

    async def __aenter__(self) -> _Connection:
        return _Connection()

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _Probe:
    def __init__(self, *, checkpointer_error: Exception | None = None, store_error: Exception | None = None) -> None:
        self.checkpointer_error = checkpointer_error
        self.store_error = store_error

    async def aget_tuple(self, _config: dict[str, object]) -> None:
        if self.checkpointer_error is not None:
            raise self.checkpointer_error

    async def aget(self, _namespace: tuple[str, ...], _key: str) -> None:
        if self.store_error is not None:
            raise self.store_error


class _DatabaseManager:
    def __init__(self, *, checkpointer: _Probe, store: _Probe) -> None:
        self.engine = _Engine()
        self._checkpointer = checkpointer
        self._store = store

    def get_checkpointer(self) -> _Probe:
        return self._checkpointer

    def get_store(self) -> _Probe:
        return self._store


def _patch_database_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkpointer: _Probe | None = None,
    store: _Probe | None = None,
) -> None:
    monkeypatch.setattr(
        health,
        "db_manager",
        _DatabaseManager(checkpointer=checkpointer or _Probe(), store=store or _Probe()),
    )


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app)


async def test_readiness_returns_ready_when_probes_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_database_manager(monkeypatch)

    response = _client().get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.parametrize(
    ("component", "probe"),
    [
        ("checkpointer", "aget_tuple"),
        ("store", "aget"),
    ],
)
async def test_readiness_returns_503_when_persistence_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    probe: str,
) -> None:
    error = RuntimeError(f"{component} probe failed")
    checkpointer = _Probe(checkpointer_error=error) if probe == "aget_tuple" else _Probe()
    store = _Probe(store_error=error) if probe == "aget" else _Probe()
    _patch_database_manager(monkeypatch, checkpointer=checkpointer, store=store)

    response = _client().get("/ready")

    assert response.status_code == 503


async def test_health_returns_healthy_when_probes_return_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_database_manager(monkeypatch)

    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "database": "connected",
        "langgraph_checkpointer": "connected",
        "langgraph_store": "connected",
    }


@pytest.mark.parametrize(
    ("component", "probe"),
    [
        ("checkpointer", "aget_tuple"),
        ("store", "aget"),
    ],
)
async def test_health_returns_503_when_persistence_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    probe: str,
) -> None:
    error = RuntimeError(f"{component} probe failed")
    checkpointer = _Probe(checkpointer_error=error) if probe == "aget_tuple" else _Probe()
    store = _Probe(store_error=error) if probe == "aget" else _Probe()
    _patch_database_manager(monkeypatch, checkpointer=checkpointer, store=store)

    response = _client().get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service unhealthy"
