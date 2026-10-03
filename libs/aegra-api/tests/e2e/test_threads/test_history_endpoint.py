import asyncio
import json

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_history_endpoint_e2e():
    """
    End-to-end test against a running server using the LangGraph SDK.
    This verifies assistant creation, run execution, join endpoint, and history retrieval.
    Requires the server to be running and accessible.
    """
    client = get_e2e_client()

    # Create an assistant (idempotent if server supports if_exists/do_nothing)
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "llm"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create response", assistant)
    assert "assistant_id" in assistant, f"Invalid assistant response: {assistant}"

    # Create a thread
    thread = await client.threads.create()
    elog("Threads.create response", thread)
    assert "thread_id" in thread, f"Invalid thread response: {thread}"
    thread_id = thread["thread_id"]

    # Initial history (likely empty)
    initial_history = await client.threads.get_history(thread_id)
    elog("Threads.get_history initial", initial_history)
    assert isinstance(initial_history, list)

    # Create a run and wait for completion using join (also validates join endpoint behavior)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "human", "content": "Hello! Tell me a short joke."}]},
    )
    elog("Runs.create response", run)
    assert "run_id" in run

    final_state = await client.runs.join(thread_id, run["run_id"])
    elog("Runs.join final_state", final_state)
    assert isinstance(final_state, dict)

    # Verify history has at least one snapshot after completing the run
    history_after = await client.threads.get_history(thread_id)
    elog("Threads.get_history after run", history_after)
    assert isinstance(history_after, list)
    assert len(history_after) >= 1, f"Expected at least one checkpoint after run; got {len(history_after)}"

    # Validate pagination with limit
    limited = await client.threads.get_history(thread_id, limit=1)
    elog("Threads.get_history limit=1", limited)
    assert isinstance(limited, list)
    assert len(limited) == 1


async def _count_stuck_thread_lookups(engine: AsyncEngine) -> int:
    # A leaked lookup session sits "idle in transaction" on its thread SELECT until returned to the pool.
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                "AND pid <> pg_backend_pid() AND state = 'idle in transaction' AND query LIKE '%FROM thread%'"
            )
        )
        return int(result.scalar_one())


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancelled_history_requests_release_db_connections() -> None:
    """Regression for #517: aborting history requests mid-flight must not leave pool connections checked out."""
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Many checkpoints make each history read slow enough to cancel mid-flight.
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": json.dumps({"delay": 0, "steps": 100})}]},
        config={"recursion_limit": 250},
    )
    await client.runs.join(thread_id, run["run_id"])

    # A read timeout fires only after the request was sent, so each one is a request aborted mid-flight.
    timeout = httpx.Timeout(30.0, read=0.01)
    async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=timeout) as http:
        results = await asyncio.gather(
            *(http.post(f"/threads/{thread_id}/history", json={"limit": 1000}) for _ in range(30)),
            return_exceptions=True,
        )
    aborted = sum(isinstance(r, httpx.ReadTimeout) for r in results)
    elog("History requests aborted mid-flight", {"aborted": aborted, "total": len(results)})
    unexpected = [
        r for r in results if not (isinstance(r, httpx.ReadTimeout) or getattr(r, "status_code", None) == 200)
    ]
    assert not unexpected, f"unexpected history results: {unexpected[:3]}"
    assert aborted > 0, "every history request finished before the read timeout; the test exercised nothing"

    # Poll: concurrent server work can hold a transaction briefly, a leaked session holds it forever.
    engine = create_async_engine(settings.db.database_url)
    try:
        leaked = await _count_stuck_thread_lookups(engine)
        for _ in range(20):
            if leaked == 0:
                break
            await asyncio.sleep(0.5)
            leaked = await _count_stuck_thread_lookups(engine)
    finally:
        await engine.dispose()
    assert leaked == 0, f"{leaked} connection(s) still checked out after aborted history requests"
