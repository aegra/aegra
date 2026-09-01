"""Tests for native LangSmith tracing session resolution."""

from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda
from langsmith import get_tracing_context, run_trees

from aegra_api.models.auth import User
from aegra_api.models.runs import LangSmithTracer
from aegra_api.observability.langsmith_tracing import (
    is_native_langsmith_tracing_enabled,
    native_langsmith_tracing_context,
    resolve_langsmith_session_name,
)
from aegra_api.services.langgraph_service import create_run_config
from aegra_api.settings import settings


def test_native_tracing_requires_explicit_flag(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings.observability, "LANGSMITH_API_KEY", None)

    assert is_native_langsmith_tracing_enabled() is True

    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", False)

    assert is_native_langsmith_tracing_enabled() is False


def test_default_project_is_the_session_name(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings.observability, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings.observability, "LANGSMITH_PROJECT", "studio-default")

    assert resolve_langsmith_session_name(None) == "studio-default"


def test_langsmith_default_project_is_used_when_unconfigured(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings.observability, "LANGSMITH_PROJECT", None)
    monkeypatch.setattr(
        "aegra_api.observability.langsmith_tracing.get_tracer_project",
        lambda: "default",
    )

    assert resolve_langsmith_session_name(None) == "default"


def test_per_run_project_overrides_the_default(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings.observability, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings.observability, "LANGSMITH_PROJECT", "studio-default")

    tracer = LangSmithTracer(project_name="studio-run")

    assert resolve_langsmith_session_name(tracer) == "studio-run"


def test_disabled_tracing_has_no_session_name(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", False)
    monkeypatch.setattr(settings.observability, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings.observability, "LANGSMITH_PROJECT", "studio-default")

    assert resolve_langsmith_session_name(LangSmithTracer(project_name="studio-run")) is None


def test_per_run_project_replicates_to_override_and_default(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings.observability, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings.observability, "LANGSMITH_PROJECT", "studio-default")

    tracer = LangSmithTracer(project_name="studio-run", example_id="example-123")

    with native_langsmith_tracing_context(tracer):
        context = get_tracing_context()

    assert context["enabled"] is True
    assert context["project_name"] == "studio-default"
    assert context["replicas"] == [
        {
            "project_name": "studio-run",
            "updates": {"reference_example_id": "example-123"},
        },
        {"project_name": "studio-default", "updates": None},
    ]


def test_default_project_does_not_create_replicas(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings.observability, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings.observability, "LANGSMITH_PROJECT", "studio-default")

    with native_langsmith_tracing_context(None):
        context = get_tracing_context()

    assert context["enabled"] is True
    assert context["project_name"] == "studio-default"
    assert context["replicas"] is None


@pytest.mark.asyncio
async def test_default_project_trace_uses_agent_run_and_thread_ids(monkeypatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings.observability, "LANGSMITH_PROJECT", "studio-default")
    client = MagicMock()
    monkeypatch.setattr(run_trees, "_CLIENT", client)
    run_id = "11111111-1111-4111-8111-111111111111"
    config = create_run_config(run_id, "thread-1", User(identity="user-1"))

    with native_langsmith_tracing_context(None):
        await RunnableLambda(lambda value: value).ainvoke({"message": "hello"}, config=config)

    created = client.create_run.call_args.kwargs
    assert str(created["id"]) == run_id
    assert created["session_name"] == "studio-default"
    assert created["extra"]["metadata"]["thread_id"] == "thread-1"
