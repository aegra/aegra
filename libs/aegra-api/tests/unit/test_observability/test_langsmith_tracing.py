"""Tests for native LangSmith tracing session resolution."""

from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda
from langsmith import get_tracing_context, run_trees

from aegra_api.models.runs import LangSmithTracer
from aegra_api.observability.langsmith_tracing import (
    native_langsmith_tracing_context,
    resolve_langsmith_session_name,
)
from aegra_api.settings import settings

EXAMPLE_ID = "11111111-1111-4111-8111-111111111111"


def _identity(value: dict[str, str]) -> dict[str, str]:
    return value


@pytest.fixture
def tracing_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", True)
    monkeypatch.setattr("aegra_api.observability.langsmith_tracing.get_tracer_project", lambda: "studio-default")


@pytest.mark.parametrize(
    ("enabled", "tracer", "expected"),
    [
        (True, None, "studio-default"),
        (True, LangSmithTracer(project_name="studio-run"), "studio-run"),
        (False, None, None),
        (False, LangSmithTracer(project_name="studio-run"), None),
    ],
)
def test_session_name_follows_the_tracing_flag_and_per_run_project(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    tracer: LangSmithTracer | None,
    expected: str | None,
) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", enabled)
    monkeypatch.setattr("aegra_api.observability.langsmith_tracing.get_tracer_project", lambda: "studio-default")

    assert resolve_langsmith_session_name(tracer) == expected


@pytest.mark.asyncio
async def test_disabled_tracing_leaves_the_context_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.observability, "LANGSMITH_TRACING", False)

    async with native_langsmith_tracing_context(LangSmithTracer(project_name="studio-run")):
        context = get_tracing_context()

    assert context["enabled"] is None
    assert context["replicas"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tracer", "expected_writes"),
    [
        (None, [("studio-default", None)]),
        (LangSmithTracer(example_id=EXAMPLE_ID), [("studio-default", EXAMPLE_ID)]),
        # A per-run project also keeps writing to the default one, so Studio still finds the run.
        (
            LangSmithTracer(project_name="studio-run", example_id=EXAMPLE_ID),
            [("studio-run", EXAMPLE_ID), ("studio-default", None)],
        ),
    ],
)
async def test_traces_export_to_every_resolved_session(
    tracing_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
    tracer: LangSmithTracer | None,
    expected_writes: list[tuple[str, str | None]],
) -> None:
    client = MagicMock()
    monkeypatch.setattr(run_trees, "_CLIENT", client)

    async with native_langsmith_tracing_context(tracer):
        await RunnableLambda(_identity).ainvoke({"message": "hello"})

    writes = [
        (call.kwargs["session_name"], str(example) if (example := call.kwargs.get("reference_example_id")) else None)
        for call in client.create_run.call_args_list
    ]
    assert writes == expected_writes
