"""Proof that a run actually reaches LangSmith — not just that Aegra accepted the request.

Opt-in: needs a real LANGSMITH_API_KEY and a server started with LANGSMITH_TRACING=true.
Uses the stress_test graph so no model provider credentials are involved.
"""

import asyncio
import json
import os
import uuid

import pytest
from langsmith import Client as LangSmithClient
from langsmith.schemas import Run as LangSmithRun
from langsmith.utils import LangSmithNotFoundError

from tests.e2e._utils import await_terminal_run, elog, get_e2e_client

EXAMPLE_ID = "11111111-1111-4111-8111-111111111111"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.getenv("LANGSMITH_API_KEY"), reason="needs a real LangSmith API key"),
]


async def _await_exported_run(langsmith: LangSmithClient, run_id: str, *, timeout: float = 60.0) -> LangSmithRun:
    """LangSmith ingests in background batches, so the trace lands after the run settles."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            return await asyncio.to_thread(langsmith.read_run, run_id)
        except LangSmithNotFoundError:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"Run {run_id} never appeared in LangSmith within {timeout}s") from None
            await asyncio.sleep(2.0)


@pytest.mark.asyncio
async def test_run_exports_to_the_requested_project_with_the_example_association() -> None:
    project = f"aegra-e2e-{uuid.uuid4().hex[:8]}"
    client = get_e2e_client()

    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 1})}]},
        langsmith_tracing={"project_name": project, "example_id": EXAMPLE_ID},
    )
    elog("Runs.create", run)

    if run["langsmith_session_name"] is None:
        pytest.skip("server was started without LANGSMITH_TRACING=true")
    assert run["langsmith_session_name"] == project

    await await_terminal_run(client, thread["thread_id"], run["run_id"])

    traced = await _await_exported_run(LangSmithClient(), run["run_id"])
    elog("LangSmith.read_run", {"id": str(traced.id), "example": str(traced.reference_example_id)})
    assert str(traced.reference_example_id) == EXAMPLE_ID
    assert traced.session_id is not None
