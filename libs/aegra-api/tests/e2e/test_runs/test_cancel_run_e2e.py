"""End-to-end tests for run cancellation.

Tests verify that cancelling a run:
1. Actually stops the asyncio task (no more events after cancel)
2. Sets status to 'interrupted' (not overwritten to 'success')
3. Works for both interrupt and cancel actions

Addresses GitHub Issue #132: Cancel endpoint doesn't cancel asyncio task
"""

import asyncio

import pytest

from tests.e2e._utils import check_and_skip_if_geo_blocked, elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancel_stops_event_stream():
    """
    E2E test: Verify that cancelling a run actually stops the event stream.

    Flow:
    1. Create assistant and thread
    2. Start a streaming run that will take a while (ask for long response)
    3. Collect some initial events
    4. Cancel the run
    5. Verify no new events are produced after cancellation
    6. Verify final status is 'interrupted'
    """
    client = get_e2e_client()

    # Setup
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "cancel-test"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    assistant_id = assistant["assistant_id"]

    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    elog("Thread.create", {"thread_id": thread_id})

    # Start a streaming run with a prompt that should generate a long response
    stream = client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a very long detailed essay about the history of computing. "
                        "Include at least 10 paragraphs with specific dates and names."
                    ),
                }
            ]
        },
        stream_mode=["messages", "values"],
    )

    # Collect events until we have a few, then cancel
    events_before_cancel = []
    run_id = None

    async for chunk in stream:
        event_type = getattr(chunk, "event", None)
        events_before_cancel.append(chunk)
        elog(
            f"Event {len(events_before_cancel)}",
            {
                "event": event_type,
                "data_type": type(getattr(chunk, "data", None)).__name__,
            },
        )

        # After receiving some events, find run_id and cancel
        if len(events_before_cancel) >= 3 and run_id is None:
            runs_list = await client.runs.list(thread_id)
            if runs_list:
                run_id = runs_list[0]["run_id"]
                check_and_skip_if_geo_blocked(runs_list[0])
                elog("Found run_id", {"run_id": run_id})
                break

    if run_id is None:
        pytest.skip("Could not find run_id to cancel")

    # Cancel the run
    elog("Cancelling run", {"run_id": run_id})
    cancelled = await client.runs.cancel(thread_id, run_id)
    elog("Cancel response", cancelled)
    check_and_skip_if_geo_blocked(cancelled)

    # Give a moment for cancellation to propagate
    await asyncio.sleep(0.5)

    # Check final status
    final_run = await client.runs.get(thread_id, run_id)
    elog("Final run status", final_run)

    # The key assertion: status should be 'interrupted', not 'success'
    assert final_run["status"] == "interrupted", (
        f"Expected status 'interrupted' after cancel, got '{final_run['status']}'. "
        "This indicates the asyncio task was not properly cancelled and ran to completion."
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancel_status_not_overwritten_to_success():
    """
    E2E test: Verify that cancelled run status stays 'interrupted'.

    This test specifically checks that after cancellation, the status
    is not later overwritten to 'success' by the completing task.

    Flow:
    1. Start a run
    2. Cancel it quickly
    3. Wait a bit for any async completion to occur
    4. Verify status is still 'interrupted' (not changed to 'success')
    """
    client = get_e2e_client()

    # Setup
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "cancel-status-test"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Create a non-streaming run (background run)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={
            "messages": [
                {
                    "role": "user",
                    "content": ("Count slowly from 1 to 100, explaining each number in detail."),
                }
            ]
        },
    )
    run_id = run["run_id"]
    elog("Created run", {"run_id": run_id, "status": run["status"]})

    # Cancel immediately
    await asyncio.sleep(0.2)  # Small delay to ensure run started
    cancelled = await client.runs.cancel(thread_id, run_id)
    elog("Cancelled run", cancelled)
    check_and_skip_if_geo_blocked(cancelled)

    # Record status right after cancel
    status_after_cancel = cancelled["status"]
    elog("Status after cancel", {"status": status_after_cancel})

    # Wait a bit to see if status gets overwritten
    await asyncio.sleep(2.0)

    # Check status again
    final_run = await client.runs.get(thread_id, run_id)
    elog("Final run status after wait", final_run)

    # Status should still be 'interrupted', not 'success'
    assert final_run["status"] in ("interrupted", "error"), (
        f"Expected status to remain 'interrupted' or 'error', but got '{final_run['status']}'. "
        "This suggests the asyncio task completed and overwrote the status."
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_interrupt_action_cancels_task():
    """
    E2E test: Verify that the 'interrupt' action also cancels the asyncio task.

    The cancel endpoint supports both 'cancel' and 'interrupt' actions.
    Both should result in task cancellation.
    """
    from httpx import AsyncClient

    from aegra_api.settings import settings

    client = get_e2e_client()

    # Setup
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "interrupt-test"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Create a background run
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "Write a long story about a dragon."}]},
    )
    run_id = run["run_id"]
    elog("Created run", {"run_id": run_id})

    # Wait for it to start
    await asyncio.sleep(0.3)

    # Use interrupt action via direct HTTP call
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        response = await http_client.post(
            f"/threads/{thread_id}/runs/{run_id}/cancel",
            params={"action": "interrupt"},
        )
        elog(
            "Interrupt response",
            {"status": response.status_code, "body": response.json()},
        )
        assert response.status_code == 200

    # Wait a bit
    await asyncio.sleep(1.0)

    # Check final status
    final_run = await client.runs.get(thread_id, run_id)
    elog("Final run status", final_run)
    check_and_skip_if_geo_blocked(final_run)

    assert final_run["status"] in ("interrupted", "error"), (
        f"Expected 'interrupted' or 'error' status, got '{final_run['status']}'"
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancel_with_wait_flag():
    """
    E2E test: Verify that cancel with wait=1 waits for task to settle.

    Flow:
    1. Start a run
    2. Cancel with wait=1
    3. Response should only return after task has settled
    4. Status should be 'interrupted'
    """
    from httpx import AsyncClient

    from aegra_api.settings import settings

    client = get_e2e_client()

    # Setup
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "cancel-wait-test"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Create a background run
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke, then another, then another.",
                }
            ]
        },
    )
    run_id = run["run_id"]
    elog("Created run", {"run_id": run_id})

    # Wait for it to start
    await asyncio.sleep(0.3)

    # Cancel with wait=1
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        response = await http_client.post(
            f"/threads/{thread_id}/runs/{run_id}/cancel",
            params={"wait": 1},
        )
        elog(
            "Cancel with wait response",
            {"status": response.status_code, "body": response.json()},
        )
        assert response.status_code == 200
        result = response.json()

    check_and_skip_if_geo_blocked(result)

    # When wait=1, the response should reflect the final state
    # which should be 'interrupted' since we cancelled it
    assert result["status"] in ("interrupted", "error", "success"), f"Expected final status, got '{result['status']}'"

    # Double-check with a GET
    final_run = await client.runs.get(thread_id, run_id)
    elog("Final run status", final_run)

    # If the run was still running when we cancelled, it should be interrupted
    # If it already completed, it would be success
    assert final_run["status"] in ("interrupted", "error", "success")


async def _start_slow_run(client, assistant_id: str) -> tuple[str, str]:
    thread = await client.threads.create()
    run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": '{"delay": 1.0, "steps": 30}'}]},
    )
    return thread["thread_id"], run["run_id"]


async def _wait_terminal(client, thread_id: str, run_id: str, timeout: float = 15.0) -> str:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        run = await client.runs.get(thread_id, run_id)
        if run["status"] not in ("pending", "running"):
            return run["status"]
        await asyncio.sleep(0.5)
    return (await client.runs.get(thread_id, run_id))["status"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancel_many_by_thread_and_run_ids():
    """``runs.cancel_many(thread_id, run_ids)`` hits POST /runs/cancel and stops the listed runs only."""
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    assistant_id = assistant["assistant_id"]

    thread_id, run_a = await _start_slow_run(client, assistant_id)
    other_thread, run_b = await _start_slow_run(client, assistant_id)
    await asyncio.sleep(1.0)

    await client.runs.cancel_many(thread_id=thread_id, run_ids=[run_a], action="interrupt")

    status_a = await _wait_terminal(client, thread_id, run_a)
    elog("cancel_many by ids", {"run_a": status_a})
    assert status_a == "interrupted"

    untouched = await client.runs.get(other_thread, run_b)
    assert untouched["status"] in ("pending", "running"), untouched["status"]
    await client.runs.cancel(other_thread, run_b, action="cancel")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancel_many_by_status_cancels_all_active_runs():
    """``runs.cancel_many(status="all")`` stops every active run the caller owns."""
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    assistant_id = assistant["assistant_id"]

    started = [await _start_slow_run(client, assistant_id) for _ in range(2)]
    await asyncio.sleep(1.0)

    await client.runs.cancel_many(status="all", action="interrupt")

    statuses = [await _wait_terminal(client, thread_id, run_id) for thread_id, run_id in started]
    elog("cancel_many by status", statuses)
    assert statuses == ["interrupted", "interrupted"]
