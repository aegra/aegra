"""Shared test helper functions and mock objects"""

from datetime import UTC, datetime
from typing import Any

from aegra_api.models import Assistant, Run, Thread


def make_assistant(
    assistant_id: str = "test-assistant-123",
    name: str = "Test Assistant",
    graph_id: str = "test-graph",
    metadata: dict[str, Any] | None = None,
    user_id: str = "test-user",
    description: str | None = None,
) -> Assistant:
    """Create a mock assistant object"""
    return Assistant(
        assistant_id=assistant_id,
        name=name,
        description=description,
        graph_id=graph_id,
        metadata_dict=metadata or {},
        user_id=user_id,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        config={},
        version=1,
    )


def make_thread(
    thread_id: str = "test-thread-123",
    status: str = "idle",
    metadata: dict[str, Any] | None = None,
    user_id: str = "test-user",
) -> Thread:
    """Create a mock thread object"""
    return Thread(
        thread_id=thread_id,
        status=status,
        metadata=metadata or {"owner": user_id},
        user_id=user_id,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def make_run(
    run_id: str = "test-run-123",
    thread_id: str = "test-thread-123",
    assistant_id: str = "test-assistant-123",
    status: str = "running",
    user_id: str = "test-user",
    metadata: dict[str, Any] | None = None,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
) -> Run:
    """Create a mock run object"""
    return Run(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=assistant_id,
        status=status,
        user_id=user_id,
        metadata=metadata or {},
        input=input_data or {"message": "test"},
        output=output_data,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class DummyRun:
    """Mock run object for testing"""

    def __init__(
        self,
        run_id: str = "test-run-123",
        thread_id: str = "test-thread-123",
        assistant_id: str = "test-assistant-123",
        status: str = "running",
        user_id: str = "test-user",
        metadata: dict[str, Any] | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
    ):
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.status = status
        self.user_id = user_id
        self.tenant_id: str | None = None
        self.metadata = metadata or {}
        self.input = input_data or {"message": "test"}
        self.output = output_data
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)


class DummyThread:
    """Mock thread object for testing"""

    def __init__(
        self,
        thread_id: str = "test-thread-123",
        status: str = "idle",
        metadata: dict[str, Any] | None = None,
        user_id: str = "test-user",
    ):
        self.thread_id = thread_id
        self.status = status
        self.metadata = metadata or {"owner": user_id}
        self.user_id = user_id
        self.tenant_id: str | None = None
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)


class DummyStoreItem:
    """Mock store item for testing"""

    def __init__(self, key: str, value: Any, namespace: tuple):
        self.key = key
        self.value = value
        self.namespace = namespace
