"""Native LangSmith tracing configuration for Studio run lookup."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langsmith import tracing_context
from langsmith.run_trees import WriteReplica
from langsmith.utils import get_tracer_project

from aegra_api.models.runs import LangSmithTracer
from aegra_api.settings import settings


def resolve_langsmith_session_name(tracer: LangSmithTracer | None) -> str | None:
    if not settings.observability.LANGSMITH_TRACING:
        return None
    return (tracer.project_name if tracer else None) or get_tracer_project()


@asynccontextmanager
async def native_langsmith_tracing_context(tracer: LangSmithTracer | None) -> AsyncIterator[None]:
    session = resolve_langsmith_session_name(tracer)
    if session is None:
        yield
        return

    # Passing project_name= would make a same-named replica drop reference_example_id,
    # so the session is expressed purely as a replica.
    default_project = get_tracer_project()
    updates = {"reference_example_id": tracer.example_id} if tracer and tracer.example_id else None
    replicas: list[WriteReplica] = [{"project_name": session, "updates": updates}]
    if session != default_project:
        replicas.append({"project_name": default_project, "updates": None})

    with tracing_context(enabled=True, replicas=replicas):
        yield
