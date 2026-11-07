"""E2E tests for the v2 ``input.inject`` command against a running server.

The Python ``langgraph-sdk`` has no ``input.inject`` method, so every command is
sent as a raw body to ``POST /threads/{id}/commands``. Runs use the
``live_input`` example graph (no LLM), which declares live input at its start,
takes injected messages before each step, and can pause for approval once;
``stress_test`` stands in for a graph that never declares it.

Skipped unless the server has ``FF_V2_EVENT_STREAMING=true`` (else 503).
"""

import asyncio
import uuid
from typing import Any

import httpx
import pytest
from langgraph_sdk import get_client

from aegra_api.settings import settings
from tests.e2e._utils import await_terminal_run, elog

_DEADLINE_SECONDS = 15.0


def _base_url() -> str:
    url = settings.app.SERVER_URL
    assert url is not None
    return url


async def _command(thread_id: str, method: str, params: dict[str, Any], command_id: int = 1) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=10.0) as http:
        resp = await http.post(
            f"/threads/{thread_id}/commands", json={"id": command_id, "method": method, "params": params}
        )
    if resp.status_code == 503:
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _assistant(graph_id: str) -> str:
    assistant = await get_client(url=_base_url()).assistants.create(graph_id=graph_id, if_exists="do_nothing")
    return assistant["assistant_id"]


async def _start(thread_id: str, graph_id: str, run_input: dict[str, Any]) -> str:
    resp = await _command(thread_id, "run.start", {"assistant_id": await _assistant(graph_id), "input": run_input})
    assert resp["type"] == "success", resp
    return resp["result"]["run_id"]


def _inject_params(message_id: str, content: str) -> dict[str, Any]:
    return {"namespace": [], "message": {"role": "user", "content": content, "id": message_id, "source": "e2e"}}


async def _inject_when_accepted(thread_id: str, message_id: str, content: str) -> dict[str, Any]:
    """Send until the run has declared live input; a just-started run has not reached its first node yet."""
    deadline = asyncio.get_running_loop().time() + _DEADLINE_SECONDS
    while True:
        resp = await _command(thread_id, "input.inject", _inject_params(message_id, content), command_id=2)
        if resp["type"] == "success" or asyncio.get_running_loop().time() >= deadline:
            return resp
        assert resp["error"] in ("not_supported", "no_such_run"), resp
        await asyncio.sleep(0.1)


async def _messages(thread_id: str) -> list[dict[str, Any]]:
    state = await get_client(url=_base_url()).threads.get_state(thread_id)
    values = state["values"]
    assert isinstance(values, dict)
    return values["messages"]


async def _new_thread() -> str:
    thread = await get_client(url=_base_url()).threads.create()
    return thread["thread_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_an_injected_message_reaches_the_running_graph() -> None:
    thread_id = await _new_thread()
    message_id = f"inj-{uuid.uuid4()}"
    run_id = await _start(
        thread_id, "live_input", {"messages": [{"role": "user", "content": "start"}], "steps": 6, "delay": 0.5}
    )

    resp = await _inject_when_accepted(thread_id, message_id, "also check the tests")
    run = await await_terminal_run(get_client(url=_base_url()), thread_id, run_id, timeout=_DEADLINE_SECONDS)
    messages = await _messages(thread_id)
    elog("input.inject reached the graph", {"resp": resp, "messages": messages})

    assert resp == {"type": "success", "id": 2, "result": {}, "meta": {"applied_through_seq": 0}}
    assert run["status"] == "success"
    injected = [message for message in messages if message.get("id") == message_id]
    assert len(injected) == 1
    assert injected[0]["type"] == "human"
    assert injected[0]["content"] == "also check the tests"
    assert injected[0]["additional_kwargs"]["source"] == "e2e"
    assert isinstance(injected[0]["additional_kwargs"]["injected_by"], str)
    # The graph drains before every step; a message it took must not come back on the next one.
    replies = [message["content"] for message in messages if message["type"] == "ai"]
    assert sum("also check the tests" in reply for reply in replies) == 1, replies


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_a_repeated_inject_is_delivered_once() -> None:
    thread_id = await _new_thread()
    message_id = f"inj-{uuid.uuid4()}"
    run_id = await _start(
        thread_id, "live_input", {"messages": [{"role": "user", "content": "start"}], "steps": 6, "delay": 0.5}
    )

    first = await _inject_when_accepted(thread_id, message_id, "only once")
    second = await _command(thread_id, "input.inject", _inject_params(message_id, "only once"), command_id=3)
    changed = await _command(thread_id, "input.inject", _inject_params(message_id, "something else"), command_id=4)
    await await_terminal_run(get_client(url=_base_url()), thread_id, run_id, timeout=_DEADLINE_SECONDS)

    assert first["type"] == "success"
    assert second["type"] == "success"
    assert changed["error"] == "invalid_argument"
    assert [message["id"] for message in await _messages(thread_id)].count(message_id) == 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_an_untaken_message_enters_the_next_run() -> None:
    thread_id = await _new_thread()
    message_id = f"inj-{uuid.uuid4()}"
    first_run = await _start(
        thread_id,
        "live_input",
        {"messages": [{"role": "user", "content": "start"}], "steps": 4, "delay": 0.5, "drain": False},
    )
    resp = await _inject_when_accepted(thread_id, message_id, "for the next run")
    await await_terminal_run(get_client(url=_base_url()), thread_id, first_run, timeout=_DEADLINE_SECONDS)
    assert resp["type"] == "success"
    assert message_id not in [message.get("id") for message in await _messages(thread_id)]

    second_run = await _start(thread_id, "live_input", {"messages": [], "steps": 1, "delay": 0, "drain": True})
    await await_terminal_run(get_client(url=_base_url()), thread_id, second_run, timeout=_DEADLINE_SECONDS)

    assert message_id in [message.get("id") for message in await _messages(thread_id)]


async def _pending_interrupt_id(thread_id: str) -> str:
    state = await get_client(url=_base_url()).threads.get_state(thread_id)
    interrupts = [item for task in state["tasks"] for item in task["interrupts"]]
    assert len(interrupts) == 1, state["tasks"]
    return interrupts[0]["id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_a_resume_after_an_interrupt_takes_but_does_not_repeat_what_the_paused_run_added() -> None:
    """The paused run took the message in a step before its interrupt, so the resume sees it again."""
    thread_id = await _new_thread()
    message_id = f"inj-{uuid.uuid4()}"
    client = get_client(url=_base_url())
    first_run = await _start(
        thread_id,
        "live_input",
        {"messages": [{"role": "user", "content": "start"}], "steps": 3, "delay": 0.5, "pause_after": 2},
    )
    resp = await _inject_when_accepted(thread_id, message_id, "also check the tests")
    paused = await await_terminal_run(client, thread_id, first_run, timeout=_DEADLINE_SECONDS)
    assert resp["type"] == "success"
    assert paused["status"] == "interrupted"
    before_pause = [message["content"] for message in await _messages(thread_id) if message["type"] == "ai"]
    assert sum("also check the tests" in reply for reply in before_pause) == 1, before_pause

    resumed = await _command(
        thread_id,
        "input.respond",
        {"interrupt_id": await _pending_interrupt_id(thread_id), "namespace": [], "response": "yes"},
        command_id=3,
    )
    assert resumed["type"] == "success", resumed
    await await_terminal_run(client, thread_id, resumed["result"]["run_id"], timeout=_DEADLINE_SECONDS)
    # The resume took the message again, so it settled: a later run must not receive it either.
    third_run = await _start(thread_id, "live_input", {"messages": [], "steps": 1, "delay": 0})
    await await_terminal_run(client, thread_id, third_run, timeout=_DEADLINE_SECONDS)

    messages = await _messages(thread_id)
    elog("resume after an interrupt", messages)
    assert [message.get("id") for message in messages].count(message_id) == 1
    replies = [message["content"] for message in messages if message["type"] == "ai"]
    assert sum("also check the tests" in reply for reply in replies) == 1, replies
    assert replies[-2:] == ["step 3: received []", "step 1: received []"], replies


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_a_run_that_does_not_declare_live_input_refuses_it() -> None:
    thread_id = await _new_thread()
    run_id = await _start(
        thread_id, "stress_test", {"messages": [{"role": "user", "content": '{"delay": 1.0, "steps": 3}'}]}
    )
    client = get_client(url=_base_url())
    deadline = asyncio.get_running_loop().time() + _DEADLINE_SECONDS
    while (await client.runs.get(thread_id, run_id))["status"] != "running":
        assert asyncio.get_running_loop().time() < deadline, "run never started"
        await asyncio.sleep(0.1)

    resp = await _command(thread_id, "input.inject", _inject_params(f"inj-{uuid.uuid4()}", "ignored?"))
    await await_terminal_run(client, thread_id, run_id, timeout=_DEADLINE_SECONDS)

    assert resp["type"] == "error"
    assert resp["error"] == "not_supported"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_inject_without_an_active_run_is_no_such_run() -> None:
    thread_id = await _new_thread()

    resp = await _command(thread_id, "input.inject", _inject_params(f"inj-{uuid.uuid4()}", "anyone there?"))

    assert resp["type"] == "error"
    assert resp["error"] == "no_such_run"
