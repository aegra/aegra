"""E2E: SDK run fields that used to be dropped now change behavior or return an error (#503)."""

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest

from tests.e2e._utils import await_terminal_run, elog, get_e2e_client

_STEPS = 2
_INPUT: dict[str, Any] = {"messages": [{"role": "user", "content": json.dumps({"delay": 0, "steps": _STEPS})}]}


async def _assistant_id(client: Any) -> str:
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    return assistant["assistant_id"]


async def _start(client: Any, method: str, thread_id: str | None, assistant_id: str, **run_kwargs: Any) -> Any:
    """Create a run through one of the three SDK entry points."""
    if method == "stream":
        return [part async for part in client.runs.stream(thread_id, assistant_id, input=_INPUT, **run_kwargs)]
    return await getattr(client.runs, method)(thread_id, assistant_id, input=_INPUT, **run_kwargs)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["create", "stream", "wait"])
async def test_reject_returns_404_and_creates_no_thread(method: str) -> None:
    client = get_e2e_client()
    assistant_id = await _assistant_id(client)
    thread_id = str(uuid4())

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _start(client, method, thread_id, assistant_id, if_not_exists="reject")

    assert exc_info.value.response.status_code == 404
    with pytest.raises(httpx.HTTPStatusError) as thread_lookup:
        await client.threads.get(thread_id)
    assert thread_lookup.value.response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_reject_runs_on_an_existing_thread() -> None:
    client = get_e2e_client()
    assistant_id = await _assistant_id(client)
    thread = await client.threads.create()

    values = await _start(client, "wait", thread["thread_id"], assistant_id, if_not_exists="reject")

    assert values["step_count"] == _STEPS


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("run_kwargs", [{}, {"if_not_exists": "create"}])
async def test_missing_thread_is_created_unless_reject_is_asked(run_kwargs: dict[str, Any]) -> None:
    client = get_e2e_client()
    assistant_id = await _assistant_id(client)
    thread_id = str(uuid4())

    values = await _start(client, "wait", thread_id, assistant_id, **run_kwargs)

    assert values["step_count"] == _STEPS
    assert (await client.threads.get(thread_id))["thread_id"] == thread_id


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_run_ignores_reject() -> None:
    client = get_e2e_client()
    assistant_id = await _assistant_id(client)

    values = await _start(client, "wait", None, assistant_id, if_not_exists="reject")

    assert values["step_count"] == _STEPS


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_delayed_start_is_rejected_and_starts_no_run() -> None:
    client = get_e2e_client()
    assistant_id = await _assistant_id(client)
    thread = await client.threads.create()

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _start(client, "create", thread["thread_id"], assistant_id, after_seconds=5)

    assert exc_info.value.response.status_code == 422
    assert "after_seconds is not supported" in exc_info.value.response.text
    assert await client.runs.list(thread["thread_id"]) == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_input_is_stored_and_returned_unchanged() -> None:
    """The run row keeps a non-object input as sent; this graph itself only takes an object."""
    client = get_e2e_client()
    assistant_id = await _assistant_id(client)
    thread = await client.threads.create()
    list_input = ["first", {"nested": 1}, 2]

    run = await client.runs.create(thread["thread_id"], assistant_id, input=list_input)
    settled = await await_terminal_run(client, thread["thread_id"], run["run_id"])
    elog("run with list input", {"status": settled["status"], "input": settled["input"]})

    assert run["input"] == list_input
    assert settled["input"] == list_input
