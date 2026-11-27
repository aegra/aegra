"""A handler filter that hides a thread must hide it on every by-id route, not only GET /threads/{id}."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from aegra_api.core import auth_scope
from aegra_api.core.auth_scope import ensure_thread_visible
from aegra_api.models.auth import User

USER = User(identity="alice", display_name="Alice")


def _maker_returning(row: str | None, seen: list[Any]) -> Callable[[], Callable[[], Any]]:
    @asynccontextmanager
    async def session() -> AsyncIterator[AsyncMock]:
        mock = AsyncMock()

        async def scalar(stmt: Any) -> str | None:
            seen.append(stmt)
            return row

        mock.scalar.side_effect = scalar
        yield mock

    return lambda: session


@pytest.mark.asyncio
@pytest.mark.parametrize("filters", [None, {}, {"metadata": {}}])
async def test_no_filter_skips_the_lookup(monkeypatch: pytest.MonkeyPatch, filters: dict[str, Any] | None) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(auth_scope, "get_session_maker", _maker_returning(None, seen))

    await ensure_thread_visible("t-1", USER, filters)

    assert seen == []


@pytest.mark.asyncio
async def test_thread_the_filter_excludes_is_a_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_scope, "get_session_maker", _maker_returning(None, []))

    with pytest.raises(HTTPException) as exc_info:
        await ensure_thread_visible("t-1", USER, {"team": "blue"})

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_lookup_scopes_by_id_owner_and_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(auth_scope, "get_session_maker", _maker_returning("t-1", seen))

    await ensure_thread_visible("t-1", USER, {"metadata": {"team": "blue"}})

    compiled = seen[0].compile(dialect=postgresql.dialect())
    sql, params = str(compiled), list(compiled.params.values())
    assert "thread.thread_id =" in sql and "t-1" in params
    assert "thread.user_id =" in sql and "alice" in params
    assert "@>" in sql and {"team": "blue"} in params
