"""Database fixtures for tests"""

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from sqlalchemy import DateTime, Insert, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import NoInspectionAvailable


def _orm_attribute_names(stmt: Insert) -> dict[str, str]:
    """Map column names to the ORM attributes a handler reads them back through.

    Assistant stores metadata in a column named ``metadata`` behind the
    ``metadata_dict`` attribute, so echoing the bound keys verbatim is not enough.
    """
    try:
        mapper = inspect(stmt.entity_description["type"])
    except (KeyError, TypeError, NoInspectionAvailable):
        return {}
    return {column.key: prop.key for prop in mapper.column_attrs for column in prop.columns}


def echo_inserted_row(stmt: Insert) -> Any:
    """Rebuild the row an ``INSERT ... RETURNING`` would hand back.

    A handler that creates atomically reads the RETURNING row rather than an
    object it built itself, so a session mock has to echo the bound values back.
    """
    params = stmt.compile(dialect=postgresql.dialect()).params
    attributes = _orm_attribute_names(stmt)
    row = {attributes.get(key, key): value for key, value in params.items()}

    # Postgres fills the timestamps the statement left out; without them a row
    # read straight back through Pydantic fails on missing created_at/updated_at.
    now = datetime.now(UTC)
    for column in stmt.table.columns:
        if column.key not in params and isinstance(column.type, DateTime):
            row.setdefault(attributes.get(column.key, column.key), now)

    return SimpleNamespace(**row)


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
