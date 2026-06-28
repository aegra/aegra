"""E2E tests driving v2 streaming through the real langgraph-sdk client.

The point of these is fidelity: they use ``client.threads.stream`` exactly
as an application (or the Vue/React ``useStream``) would, so passing them
proves wire compatibility with the stock SDK, not just our own wire shape.

Skipped unless the server has ``FF_V2_EVENT_STREAMING=true`` (else 503).
Uses the ``stress_test`` graph (no LLM) so the run is hermetic.
"""

import asyncio
import json

import httpx
import pytest
from langgraph_sdk import get_client

from aegra_api.settings import settings
from tests.e2e._utils import elog


def _base_url() -> str:
    url = settings.app.SERVER_URL
    assert url is not None
    return url


async def _v2_enabled() -> bool:
    """True if the server has v2 streaming on (a thread command returns non-503)."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=10.0) as http:
        client = get_client(url=_base_url())
        thread = await client.threads.create()
        resp = await http.post(
            f"/threads/{thread['thread_id']}/commands",
            json={"id": 0, "method": "run.start", "params": {}},
        )
        return resp.status_code != 503


async def _ensure_assistant() -> str:
    client = get_client(url=_base_url())
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    return assistant["assistant_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_thread_stream_run_start_and_events() -> None:
    """The stock SDK starts a run and receives v2 events over the thread stream."""
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    assistant_id = await _ensure_assistant()
    client = get_client(url=_base_url())

    methods: list[str] = []
    lifecycle_events: list[str] = []
    async with client.threads.stream(assistant_id=assistant_id) as ts:
        await ts.run.start(input={"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 1})}]})
        async for event in ts.events:
            method = event.get("method")
            methods.append(method)
            if method == "lifecycle":
                lifecycle_events.append(event["params"]["data"]["event"])
            if "completed" in lifecycle_events or "failed" in lifecycle_events:
                break

    elog("sdk thread stream methods", methods)
    assert "lifecycle" in methods, f"no lifecycle event received; got {methods}"
    assert "completed" in lifecycle_events, f"run did not complete; lifecycle={lifecycle_events}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_receives_values_events() -> None:
    """The SDK receives values-channel events carrying the run's state."""
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    assistant_id = await _ensure_assistant()
    client = get_client(url=_base_url())

    value_payloads: list[dict] = []
    async with client.threads.stream(assistant_id=assistant_id) as ts:
        await ts.run.start(input={"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 1})}]})
        async for event in ts.events:
            if event.get("method") == "values":
                value_payloads.append(event["params"]["data"])
            if event.get("method") == "lifecycle" and event["params"]["data"]["event"] in ("completed", "failed"):
                break

    elog("sdk values events", value_payloads)
    assert value_payloads, "no values events received"
    assert any("messages" in payload for payload in value_payloads)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_raw_wire_body_matches_sdk_contract() -> None:
    """The stream endpoint accepts the exact body the SDK sends: {channels} only."""
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    client = get_client(url=_base_url())
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # No run started and no run_id in the body — must open (200), not 4xx.
    async with (
        httpx.AsyncClient(base_url=_base_url(), timeout=15.0) as http,
        http.stream("POST", f"/threads/{thread_id}/stream/events", json={"channels": ["lifecycle"]}) as resp,
    ):
        assert resp.status_code == 200, f"SDK-shaped body rejected: {resp.status_code}"
        await resp.aclose()


async def _ensure_graph(graph_id: str) -> str:
    client = get_client(url=_base_url())
    assistant = await client.assistants.create(graph_id=graph_id, if_exists="do_nothing")
    return assistant["assistant_id"]


def _content_block_events(events: list[dict]) -> set[str]:
    """The set of message content-block lifecycle events seen on the stream."""
    seen: set[str] = set()
    for event in events:
        if event.get("method") != "messages":
            continue
        data = event["params"]["data"]
        if isinstance(data, dict) and isinstance(data.get("event"), str):
            seen.add(data["event"])
    return seen


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_tool_agent_streams_content_blocks_and_tool_calls() -> None:
    """A tool-calling agent streams the full content-block lifecycle incl. tool-call blocks.

    This is the path that was invisible before going native — tool calls now
    arrive as ``content-block-*`` events on the messages channel.
    """
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    assistant_id = await _ensure_graph("stress_tool_agent")
    client = get_client(url=_base_url())

    events: list[dict] = []
    tool_call_seen = False
    async with client.threads.stream(assistant_id=assistant_id) as ts:
        await ts.run.start(input={"messages": [{"role": "user", "content": "Process steps 1 and 2."}]})
        async for event in ts.events:
            events.append(event)
            data = event.get("params", {}).get("data") or {}
            if isinstance(data, dict) and "tool_call" in json.dumps(data):
                tool_call_seen = True
            if event.get("method") == "lifecycle" and data.get("event") in ("completed", "failed"):
                break

    blocks = _content_block_events(events)
    elog("tool agent content-block events", sorted(blocks))
    assert "message-start" in blocks
    assert "content-block-delta" in blocks
    assert "message-finish" in blocks
    assert tool_call_seen, "no tool-call content reached the stream"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_hitl_interrupt_surfaces_on_input_channel_and_resumes() -> None:
    """A HITL interrupt surfaces as input.requested; input.respond resumes the run.

    This is the path that was broken before going native — the SDK could not
    learn the interrupt id, so it could not resume.
    """
    if not await _v2_enabled():
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

    assistant_id = await _ensure_graph("agent_hitl")
    client = get_client(url=_base_url())

    input_requested: list[dict] = []
    resumed = False
    lifecycle_after_resume: list[str] = []
    async with client.threads.stream(assistant_id=assistant_id) as ts:
        # A search request makes the agent call a tool, which the graph gates
        # behind a human-approval interrupt.
        await ts.run.start(
            input={"messages": [{"role": "user", "content": "Search the web for the latest LangGraph release."}]}
        )
        async for event in ts.events:
            method = event.get("method")
            data = event.get("params", {}).get("data") or {}
            if method == "input.requested" and not resumed:
                input_requested.append(data)
                # Resume on the SAME open stream — the SDK does not reopen it.
                # Proves the session keeps the stream alive across the run gap
                # and that resume works without re-supplying an assistant.
                await _resume_via_sdk(ts, data["interrupt_id"])
                resumed = True
                continue
            if resumed and method == "lifecycle":
                lifecycle_after_resume.append(data.get("event"))
                if data.get("event") in ("completed", "failed"):
                    break

    elog("hitl input.requested", input_requested)
    elog("hitl lifecycle after resume", lifecycle_after_resume)
    assert input_requested, "interrupt did not surface on the input channel"
    assert isinstance(input_requested[0].get("interrupt_id"), str)
    # value (not payload) is the SDK's InterruptPayload field.
    assert "value" in input_requested[0], "interrupt value missing from input.requested"
    assert "completed" in lifecycle_after_resume, (
        f"resume did not run to completion on the same stream; got {lifecycle_after_resume}"
    )


async def _resume_via_sdk(ts: object, interrupt_id: str) -> None:
    """Call ``ts.run.respond`` once the SDK's lifecycle watcher has registered the
    interrupt. The watcher runs on a separate SSE, so the main stream can surface
    ``input.requested`` a beat before ``ts.interrupts`` is populated."""
    for _ in range(50):
        if any(p.get("interrupt_id") == interrupt_id for p in ts.interrupts):  # type: ignore[attr-defined]
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError(f"SDK never registered interrupt {interrupt_id}: {ts.interrupts!r}")  # type: ignore[attr-defined]
    await ts.run.respond({"action": "approve"}, interrupt_id=interrupt_id)  # type: ignore[attr-defined]
