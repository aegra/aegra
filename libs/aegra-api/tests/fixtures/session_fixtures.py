"""Shared session fixtures for testing"""

from typing import Any, Self

from sqlalchemy import Insert

from tests.fixtures.database import (
    DummyScalarResult,
    DummySessionBase,
    override_get_session_dep,
)


class BasicSession(DummySessionBase):
    """Basic session with minimal functionality"""

    def add(self, obj: Any) -> None:
        """Mock add method"""
        pass

    async def commit(self) -> None:
        """Mock commit method"""
        pass

    async def refresh(self, obj: Any) -> None:
        """Mock refresh method"""
        pass


class ThreadSession(BasicSession):
    """Session test double for thread operations."""

    def __init__(self: Self, threads: list[Any] | None = None, count: int | None = None) -> None:
        """Initialize ThreadSession with optional threads list and count override."""
        super().__init__()
        self.threads = threads or []
        self.count = count

    def _filter_threads(self: Self, stmt: Any = None) -> list[Any]:
        """Filter threads based on whereclause conditions in the statement."""
        if stmt is None or not hasattr(stmt, "whereclause") or stmt.whereclause is None:
            return list(self.threads)

        clauses = getattr(stmt.whereclause, "clauses", [stmt.whereclause])
        filtered = list(self.threads)
        for clause in clauses:
            left_name = getattr(getattr(clause, "left", None), "key", None) or str(getattr(clause, "left", ""))
            right_val = getattr(getattr(clause, "right", None), "value", None)

            if "status" in left_name and right_val is not None:
                filtered = [t for t in filtered if getattr(t, "status", None) == right_val]
            elif "metadata_json" in left_name and isinstance(right_val, dict):
                filtered = [
                    t
                    for t in filtered
                    if isinstance(getattr(t, "metadata_json", None), dict)
                    and all(getattr(t, "metadata_json", {}).get(k) == v for k, v in right_val.items())
                ]
            elif "user_id" in left_name and right_val is not None:
                filtered = [t for t in filtered if getattr(t, "user_id", None) == right_val]
        return filtered

    async def scalar(self: Self, stmt: Any = None) -> Any:
        """Return scalar count or value, evaluating whereclause filters if present."""
        if self.count is not None:
            return self.count
        filtered = self._filter_threads(stmt)
        return len(filtered)

    async def scalars(self: Self, stmt: Any = None) -> Any:
        """Return scalar results, evaluating whereclause filters if present."""
        if isinstance(stmt, Insert):
            return await super().scalars(stmt)
        filtered = self._filter_threads(stmt)
        return DummyScalarResult(filtered)


class RunSession(BasicSession):
    """Session for run operations"""

    def __init__(self, runs: list[Any] | None = None):
        super().__init__()
        self.runs = runs or []

    async def scalars(self, stmt: Any) -> Any:
        """Mock scalars method for run queries"""

        class Result:
            def all(self) -> list[Any]:
                return self.runs

        return Result()


def create_session_fixture(session_class: type = BasicSession, **kwargs):
    """Create a session fixture with the specified class and parameters"""

    def _session():
        return session_class(**kwargs)

    return _session


def override_session_dependency(app, session_class: type = BasicSession, **kwargs):
    """Override the session dependency with a mock session"""
    from aegra_api.core.orm import get_session as core_get_session

    def session_factory():
        return session_class(**kwargs)

    app.dependency_overrides[core_get_session] = override_get_session_dep(session_factory)
