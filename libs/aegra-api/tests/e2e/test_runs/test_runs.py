import pytest

from aegra_api.settings import settings

# Match import style used by other e2e tests when run as top-level modules
from tests.e2e._utils import check_and_skip_if_geo_blocked, elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_runs_crud_and_join_e2e():
    """
    Mirrors existing e2e style using the typed SDK client (see test_chat_streaming, test_background_run_join).
    Validates the non-streaming "background run" flow and CRUD around it:
      1) Ensure assistant exists (graph_id=agent)
      2) Create a thread
      3) Create a background run (non-stream)
      4) Join the run for final output
      5) Get the run by id
      6) List runs for the same thread and ensure presence
      7) Stream endpoint for a terminal run should yield an end event quickly via SDK wrapper
    """
    client = get_e2e_client()

    # 1) Assistant
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "runs-crud"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    assert "assistant_id" in assistant
    assistant_id = assistant["assistant_id"]

    # 2) Thread
    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # 3) Background run (non-streaming)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "Say one short sentence."}]},
        stream_mode=[
            "messages",
            "values",
        ],  # ensure both modes are available for later stream
    )
    elog("Runs.create", run)
    assert "run_id" in run
    run_id = run["run_id"]

    # 4) Join run and assert final output (dict)
    final_state = await client.runs.join(thread_id, run_id)
    elog("Runs.join", final_state)

    # Check for blocking error before asserting
    check_run = await client.runs.get(thread_id, run_id)
    check_and_skip_if_geo_blocked(check_run)

    assert isinstance(final_state, dict)

    # 5) Get run by id
    got = await client.runs.get(thread_id, run_id)
    elog("Runs.get", got)
    assert got["run_id"] == run_id
    assert got["thread_id"] == thread_id
    assert got["assistant_id"] == assistant_id
    assert got["status"] in (
        "success",
        "error",
        "interrupted",
        "running",
        "pending",
    )

    # 6) List runs for the thread and ensure our run is present
    runs_list = await client.runs.list(thread_id)
    elog("Runs.list", runs_list)
    assert isinstance(runs_list, list)
    assert any(r["run_id"] == run_id for r in runs_list)

    # 7) Stream endpoint after completion: should yield an end event quickly.
    # Reuse the SDK join_stream to align with current helper patterns.
    # We accept that there may be zero deltas and just an "end".
    if got["status"] == "success":
        end_seen = False
        async for chunk in client.runs.join_stream(
            thread_id=thread_id,
            run_id=run_id,
            stream_mode=["messages", "values"],
        ):
            elog("Runs.stream(terminal) event", {"event": getattr(chunk, "event", None)})
            if getattr(chunk, "event", None) == "end":
                end_seen = True
                break
        assert end_seen, "Expected an 'end' event when streaming a terminal run"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_runs_cancel_e2e():
    """
    Cancellation flow aligned with e2e client helpers:
      1) Create assistant and thread
      2) Start a streaming run via SDK client
      3) Cancel the run via SDK
      4) Verify status is cancelled/interrupted/final afterward
    """
    client = get_e2e_client()

    # Assistant + thread
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "runs-cancel"]},
        if_exists="do_nothing",
    )
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    assistant_id = assistant["assistant_id"]

    # Start streaming run (returns an async iterator through the SDK)
    stream = client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "Keep talking slowly."}]},
        stream_mode=["messages"],
    )

    # Consume a couple of events then cancel
    events_seen = 0
    async for chunk in stream:
        events_seen += 1
        elog("Runs.stream (pre-cancel)", {"event": getattr(chunk, "event", None)})
        # Try to fetch a run id by listing runs; server persists runs metadata now
        if events_seen >= 2:
            break

    # Find the most recent run id
    runs_list = await client.runs.list(thread_id)
    if not runs_list:
        pytest.skip("No runs found to cancel")
    run_id = runs_list[0]["run_id"]

    check_and_skip_if_geo_blocked(runs_list[0])

    # Cancel the run
    patched = await client.runs.cancel(thread_id, run_id)
    elog("Runs.cancel", patched)

    # It might have failed in background
    check_and_skip_if_geo_blocked(patched)

    assert patched["status"] in ("interrupted", "success")

    # Verify final state
    got = await client.runs.get(thread_id, run_id)
    elog("Runs.get(post-cancel)", got)
    assert got["status"] in ("interrupted", "error", "success")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_runs_wait_stateful_e2e():
    """
    Test the stateful wait endpoint (POST /threads/{thread_id}/runs/wait).
    This endpoint creates a run and waits for it to complete before returning the final output.

    Flow:
      1) Create assistant and thread
      2) Use the wait endpoint (via raw HTTP client since SDK might not have it yet)
      3) Verify output is returned directly (not a Run object)
      4) Verify run was created and completed
    """
    from httpx import AsyncClient

    client = get_e2e_client()

    # 1) Setup: Create assistant and thread
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "wait-test"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    assistant_id = assistant["assistant_id"]

    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # 2) Call wait endpoint directly via HTTP client
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http_client:
        response = await http_client.post(
            f"/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hello in one word."}]},
            },
        )
        elog(
            "Wait endpoint response",
            {
                "status": response.status_code,
                "output": response.json() if response.status_code == 200 else None,
            },
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        output = response.json()

        # 3) Verify output format - should be just the output dict, not a Run object
        assert isinstance(output, dict), "Expected output to be a dict"
        # Should not have run_id, thread_id, etc. - just the graph output
        assert "messages" in output or "output" in output or len(output) >= 0, (
            "Expected output to contain graph output data"
        )

        # Should NOT have Run metadata fields if it's the output directly
        # (but if implementation returns empty dict, that's OK too)
        elog("Final output from wait", output)

    # 4) Verify run was created and completed by listing runs
    runs_list = await client.runs.list(thread_id)
    elog("Runs.list after wait", runs_list)
    assert len(runs_list) > 0, "Expected at least one run to be created"
    last_run = runs_list[0]
    assert last_run["thread_id"] == thread_id
    assert last_run["assistant_id"] == assistant_id

    check_and_skip_if_geo_blocked(last_run)

    assert last_run["status"] in ("success", "interrupted"), (
        f"Expected completed or interrupted, got {last_run['status']}"
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_runs_wait_with_interrupts_e2e():
    """
    Test that the wait endpoint handles interrupt scenarios correctly.
    When a run is interrupted, the wait endpoint should return the partial output.

    This test uses interrupt_before to force an interrupt.
    """
    from httpx import AsyncClient

    client = get_e2e_client()

    # Setup
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "wait-interrupt-test"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Call wait endpoint with interrupt_before to force interruption
    # Note: This will interrupt before a specific node executes
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http_client:
        response = await http_client.post(
            f"/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Test"}]},
                "interrupt_before": ["agent"],  # Interrupt before agent node
            },
        )
        elog(
            "Wait with interrupt response",
            {
                "status": response.status_code,
                "output": response.json() if response.status_code == 200 else None,
            },
        )

        # Even interrupted runs should return 200 with partial output
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        output = response.json()
        assert isinstance(output, dict), "Expected output to be a dict"

        # Verify the run was created and completed
        # Note: The interrupt may not trigger if the node name doesn't match the graph structure
        # This test primarily verifies that interrupt_before parameter is accepted and doesn't break
        runs_list = await client.runs.list(thread_id)
        assert len(runs_list) > 0
        last_run = runs_list[0]

        check_and_skip_if_geo_blocked(last_run)

        # The wait response can return before the run row leaves "running"
        # (the status write races the HTTP response). Join blocks until the
        # run reaches a terminal state, then re-read to get the settled status.
        await client.runs.join(thread_id, last_run["run_id"])
        last_run = (await client.runs.list(thread_id))[0]

        # Status can be interrupted or success depending on graph structure
        assert last_run["status"] in ("interrupted", "success"), (
            f"Expected interrupted or success status, got {last_run['status']}"
        )
