"""Database fixtures for tests"""

from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

from sqlalchemy import Insert
from sqlalchemy.dialects import postgresql


def echo_inserted_row(stmt: Insert) -> Any:
    """Rebuild the row an ``INSERT ... RETURNING`` would hand back.

    A handler that creates atomically reads the RETURNING row rather than an
    object it built itself, so a session mock has to echo the bound values back.
    """
    params = stmt.compile(dialect=postgresql.dialect()).params
    return SimpleNamespace(**params)


class DummyScalarResult:
    """Minimal emulation of SQLAlchemy's ScalarResult"""

    def __init__(self, rows: list[Any] | None = None):
        self._rows = rows or []

    def all(self) -> list[Any]:
        return self._rows

    def first(self) -> Any:
        return self._rows[0] if self._rows else None


class DummySessionBase:
    """Minimal emulation of SQLAlchemy AsyncSession for testing

    Override scalar/scalars/commit/refresh in subclasses/fixtures to return
    appropriate rows for a test. By default, an INSERT is treated as having
    inserted its row and anything else returns empty data.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, _):
        """AsyncSession.add is sync in SQLAlchemy"""
        return None

    async def commit(self):
        return None

    async def refresh(self, _obj):
        return None

    async def scalar(self, _stmt):
        return None

    async def scalars(self, stmt=None):
        if isinstance(stmt, Insert):
            return DummyScalarResult([echo_inserted_row(stmt)])
        return DummyScalarResult()


def override_get_session_dep(
    session_factory: Callable[[], DummySessionBase],
) -> Callable[[], AsyncIterator[DummySessionBase]]:
    """Create a dependency override for get_session"""

    async def _dep():
        yield session_factory()

    return _dep
