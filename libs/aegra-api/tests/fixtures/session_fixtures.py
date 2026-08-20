"""Shared session fixtures for testing"""

from typing import Any

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
    """Session for thread operations"""

    def __init__(self, threads: list[Any] | None = None):
        super().__init__()
        self.threads = threads or []

    async def scalars(self, stmt: Any = None) -> Any:
        # Inserts go to the base, which echoes the RETURNING row create paths read.
        if isinstance(stmt, Insert):
            return await super().scalars(stmt)
        return DummyScalarResult(self.threads)


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
