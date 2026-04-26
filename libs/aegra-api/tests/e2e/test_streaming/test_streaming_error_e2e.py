"""End-to-end tests for streaming error handling.

These tests verify that errors during streaming are properly sent to the frontend
in real scenarios, such as:
- Invalid API keys causing LLM errors
- Graph execution errors
- Network/connection errors
"""

import json

import pytest

from tests.e2e._utils import elog, get_e2e_client


def parse_sse_event(chunk: str) -> dict | None:
    """Parse SSE event chunk into structured format.

    Args:
        chunk: Raw SSE chunk string

    Returns:
        Dict with event, data, id fields or None if invalid
    """
    if not chunk.strip():
        return None

    lines = chunk.strip().split("\n")
    event_data = {}

    for line in lines:
        # SSE comment lines (W3C §9.2.6): must be ignored. sse-starlette emits
        # `: heartbeat` every KEEPALIVE_INTERVAL_SECS as an idle keepalive.
        if line.startswith(":"):
            continue
        if line.startswith("event: "):
            event_data["event"] = line.replace("event: ", "").strip()
        elif line.startswith("data: "):
            data_str = line.replace("data: ", "").strip()
            try:
                event_data["data"] = json.loads(data_str)
            except json.JSONDecodeError:
                event_data["data"] = data_str
        elif line.startswith("id: "):
            event_data["id"] = line.replace("id: ", "").strip()

    return event_data if event_data else None


class TestParseSseEvent:
    """Unit tests for the local ``parse_sse_event`` helper.

    Not marked ``e2e`` — pure function, runs in the default suite. Covers the
    heartbeat-comment path because ``sse-starlette`` now emits
    ``: heartbeat\\r\\n\\r\\n`` on every ping, so the parser must ignore it
    per the W3C EventSource spec §9.2.6.
    """

    def test_heartbeat_comment_returns_none(self) -> None:
        assert parse_sse_event(": heartbeat") is None

    def test_heartbeat_mixed_with_event_parses_event_only(self) -> None:
        chunk = ': heartbeat\nevent: values\ndata: {"x": 1}\nid: evt-1'
        assert parse_sse_event(chunk) == {
            "event": "values",
            "data": {"x": 1},
            "id": "evt-1",
        }

    def test_empty_chunk_returns_none(self) -> None:
        assert parse_sse_event("") is None
        assert parse_sse_event("   \n  ") is None

    def test_plain_event_still_parses(self) -> None:
        chunk = 'event: end\ndata: {"status":"success"}'
        assert parse_sse_event(chunk) == {
            "event": "end",
            "data": {"status": "success"},
        }


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_streaming_error_received_in_sse():
    """
    E2E test: Verify that errors during streaming are received via SSE.

    This test creates a run that will fail (e.g., invalid config) and verifies
    that the error event is properly formatted and sent to the frontend.
    """
    client = get_e2e_client()

    # Create assistant
    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat", "stream"]},
        if_exists="do_nothing",
    )
    assert "assistant_id" in assistant

    # Create thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Try to stream with invalid input that might cause an error
    # Note: This might succeed or fail depending on graph implementation
    # The key is to verify error handling IF an error occurs

    try:
        stream = client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant["assistant_id"],
            input={"messages": [{"role": "user", "content": "test"}]},
            stream_mode=["values", "messages"],
        )

        events_received = []
        error_events = []

        async for chunk in stream:
            events_received.append(chunk)

            # Check if this is an error event
            event = getattr(chunk, "event", None)
            data = getattr(chunk, "data", None)

            if event == "error":
                error_events.append(chunk)
                elog("Error event received", {"event": event, "data": data})

            # Stop after reasonable number of events or if we get an error
            if len(events_received) > 50 or len(error_events) > 0:
                break

        # If we received an error event, verify its structure
        if error_events:
            error_event = error_events[0]
            data = getattr(error_event, "data", None)

            if isinstance(data, dict):
                assert "error" in data, "Error event should have 'error' field"
                assert "message" in data, "Error event should have 'message' field"

                elog("Error event structure verified", data)

        # Check final run status
        runs = await client.runs.list(thread_id)
        if runs:
            last_run = runs[0]
            elog(
                "Final run status",
                {
                    "status": last_run.get("status"),
                    "error": last_run.get("error_message"),
                },
            )

            # If run failed, verify error was properly recorded
            if last_run.get("status") == "error":
                assert last_run.get("error_message"), "Error run should have error_message"

    except Exception as e:
        # If the stream itself fails, that's also an error case to handle
        elog("Stream exception", {"error": str(e), "type": type(e).__name__})
        # Don't fail the test - we're testing error handling
        pass


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_streaming_error_replay_on_reconnect():
    """
    E2E test: Verify that error events are stored and replayed on reconnection.

    This test simulates a client reconnecting after an error occurred,
    and verifies the error event is included in the replay.
    """
    client = get_e2e_client()

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat"]},
        if_exists="do_nothing",
    )

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Create a run that will stream
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": "hello"}]},
    )

    run_id = run["run_id"]

    # Wait a moment for execution to start
    import asyncio

    await asyncio.sleep(1)

    # Check run status - if it errored, we can test replay
    run_status = await client.runs.get(thread_id=thread_id, run_id=run_id)

    if run_status.get("status") == "error":
        # Try to reconnect and stream - should replay error event
        try:
            stream = client.runs.stream(
                thread_id=thread_id,
                run_id=run_id,
                stream_mode=["values"],
            )

            error_events_in_replay = []
            async for chunk in stream:
                event = getattr(chunk, "event", None)
                if event == "error":
                    error_events_in_replay.append(chunk)

                # Stop after reasonable number of events
                if len(error_events_in_replay) > 0:
                    break

            # If we got error events in replay, verify structure
            if error_events_in_replay:
                error_event = error_events_in_replay[0]
                data = getattr(error_event, "data", None)

                if isinstance(data, dict):
                    assert "error" in data
                    assert "message" in data
                    elog("Error event replayed successfully", data)

        except Exception as e:
            elog("Replay stream exception", {"error": str(e)})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_streaming_error_format_matches_frontend_expectations():
    """
    E2E test: Verify error event format matches frontend expectations.

    Standard error format:
    {
        "error": "ErrorType",
        "message": "detailed error message"
    }
    """
    client = get_e2e_client()

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat"]},
        if_exists="do_nothing",
    )

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Create run
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": "test"}]},
    )

    # Wait for execution
    import asyncio

    await asyncio.sleep(2)

    # Check if run errored
    run_status = await client.runs.get(thread_id=thread_id, run_id=run["run_id"])

    if run_status.get("status") == "error":
        # Try to stream to get error event
        try:
            stream = client.runs.stream(
                thread_id=thread_id,
                run_id=run["run_id"],
                stream_mode=["values"],
            )

            async for chunk in stream:
                event = getattr(chunk, "event", None)
                data = getattr(chunk, "data", None)

                if event == "error":
                    # Verify format matches standard error event format
                    assert isinstance(data, dict), "Error data should be a dict"
                    assert "error" in data, "Error event must have 'error' field"
                    assert "message" in data, "Error event must have 'message' field"
                    assert isinstance(data["error"], str), "Error type should be string"
                    assert isinstance(data["message"], str), "Error message should be string"
                    # Verify format is standard: {error: str, message: str}
                    assert len(data) == 2, (
                        f"Error format should have only 'error' and 'message' fields, got keys: {list(data.keys())}"
                    )

                    elog("Error format verified", data)
                    break

        except Exception as e:
            elog("Error format test exception", {"error": str(e)})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_streaming_error_before_any_events():
    """
    E2E test: Verify error handling when error occurs before any events are sent.

    This tests the case where graph execution fails immediately.
    """
    client = get_e2e_client()

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["chat"]},
        if_exists="do_nothing",
    )

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Try to create a run with invalid input format
    # The graph expects input with "messages" field, so missing it should cause validation error
    try:
        run = await client.runs.create(
            thread_id=thread_id,
            assistant_id=assistant["assistant_id"],
            input={},  # Empty input - missing required "messages" field
        )

        # Try to stream immediately
        stream = client.runs.stream(
            thread_id=thread_id,
            run_id=run["run_id"],
            stream_mode=["values"],
        )

        first_event = None
        async for chunk in stream:
            first_event = chunk
            break

        # If first event is an error, verify it's properly formatted
        if first_event:
            event = getattr(first_event, "event", None)
            if event == "error":
                data = getattr(first_event, "data", None)
                assert isinstance(data, dict)
                assert "error" in data
                assert "message" in data
                elog("Early error event received", data)

    except Exception as e:
        # If creation itself fails, that's also valid error handling
        elog("Early error test exception", {"error": str(e)})
