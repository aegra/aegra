"""Native LangSmith tracing configuration for Studio run lookup."""

from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import tracing_context
from langsmith.run_trees import WriteReplica
from langsmith.utils import get_tracer_project

from aegra_api.models.runs import LangSmithTracer
from aegra_api.settings import settings


def is_native_langsmith_tracing_enabled() -> bool:
    return settings.observability.LANGSMITH_TRACING


def resolve_langsmith_session_name(tracer: LangSmithTracer | None) -> str | None:
    if not is_native_langsmith_tracing_enabled():
        return None
    if tracer is not None and tracer.project_name:
        return tracer.project_name
    return settings.observability.LANGSMITH_PROJECT or get_tracer_project()


@contextmanager
def native_langsmith_tracing_context(tracer: LangSmithTracer | None) -> Iterator[None]:
    if not is_native_langsmith_tracing_enabled():
        yield
        return

    default_project = settings.observability.LANGSMITH_PROJECT or get_tracer_project()
    if tracer is not None and tracer.project_name:
        updates = {"reference_example_id": tracer.example_id} if tracer.example_id else None
        replicas: list[WriteReplica] = [
            {"project_name": tracer.project_name, "updates": updates},
            {"project_name": default_project, "updates": None},
        ]
        with tracing_context(enabled=True, project_name=default_project, replicas=replicas):
            yield
        return

    if tracer is not None and tracer.example_id:
        with (
            tracing_context(enabled=True, project_name=default_project),
            tracing_v2_enabled(project_name=default_project, example_id=tracer.example_id),
        ):
            yield
        return

    with tracing_context(enabled=True, project_name=default_project):
        yield
