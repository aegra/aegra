"""E2E: Thread responses carry the latest ``values`` and ``interrupts`` (issue #648)."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from langgraph_sdk.client import LangGraphClient

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client


def _stress_input(**config: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": json.dumps({"steps": 1, "delay": 0, **config})}]}


async def _poll_run(client: LangGraphClient, thread_id: str, run_id: str) -> dict[str, Any]:
    for _ in range(100):
        run = await client.runs.get(thread_id=thread_id, run_id=run_id)
        if run["status"] not in ("pending", "running"):
            return dict(run)
        await asyncio.sleep(0.1)
    raise AssertionError(f"Run {run_id} did not finish")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_deepagents_async_subagent_check_reads_result_from_thread_values() -> None:
    """Replays deepagents' start_async_task / check_async_task calls, the flow broken in #648."""
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")

    # start_async_task: threads.create + runs.create, no wait.
    thread = await client.threads.create()
    run = await client.runs.create(
        thread_id=thread["thread_id"], assistant_id=assistant["assistant_id"], input=_stress_input()
    )

    # check_async_task: runs.get, then on success read the answer from threads.get()["values"].
    finished = await _poll_run(client, thread["thread_id"], run["run_id"])
    assert finished["status"] == "success"
    fetched = await client.threads.get(thread_id=thread["thread_id"])
    elog("threads.get as deepagents reads it", fetched)
    messages = (fetched.get("values") or {}).get("messages", [])
    assert messages, "deepagents would report '(completed with no output messages)'"
    assert json.loads(messages[-1]["content"])["status"] == "completed"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_get_and_search_return_latest_values_after_run() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    assert thread["values"] is None
    assert thread["interrupts"] == {}

    await client.runs.wait(
        thread_id,
        assistant["assistant_id"],
        input=_stress_input(),
    )

    fetched = await client.threads.get(thread_id)
    elog("threads.get after run", fetched)
    state = await client.threads.get_state(thread_id)
    assert fetched["values"] == state["values"]
    assert fetched["values"]["messages"][-1]["type"] == "ai"
    assert fetched["interrupts"] == {}

    [searched] = [t for t in await client.threads.search(limit=100) if t["thread_id"] == thread_id]
    assert searched["values"] == state["values"]

    patched = await client.threads.update(thread_id, metadata={"label": "checked"})
    assert patched["values"] == state["values"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_threads_omits_state_values() -> None:
    """GET /threads is unpaginated, so it lists threads without their cached state."""
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    await client.runs.wait(thread["thread_id"], assistant["assistant_id"], input=_stress_input())

    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.get(f"{settings.app.SERVER_URL}/threads")

    assert resp.status_code == 200
    [listed] = [t for t in resp.json()["threads"] if t["thread_id"] == thread["thread_id"]]
    assert listed["status"] == "idle"
    assert "values" not in listed


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_failed_run_keeps_previous_thread_values() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    await client.runs.wait(thread_id, assistant["assistant_id"], input=_stress_input())
    before = await client.threads.get(thread_id)

    run = await client.runs.create(thread_id, assistant["assistant_id"], input=_stress_input(fail=True))
    finished = await _poll_run(client, thread_id, run["run_id"])

    assert finished["status"] == "error"
    after = await client.threads.get(thread_id)
    elog("threads.get after failed run", after)
    assert after["status"] == "error"
    assert after["values"] == before["values"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_interrupts_are_keyed_by_task_and_clear_on_resume() -> None:
    client = get_e2e_client()
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    await client.runs.wait(thread_id, "subgraph_hitl_agent", input={"foo": "Test value."})

    interrupted = await client.threads.get(thread_id)
    elog("threads.get while interrupted", interrupted)
    assert interrupted["status"] == "interrupted"
    [task_interrupts] = interrupted["interrupts"].values()
    assert [i["value"] for i in task_interrupts] == ["Provide value:"]
    assert task_interrupts[0]["id"]

    await client.runs.wait(thread_id, "subgraph_hitl_agent", command={"resume": " resumed"})

    resumed = await client.threads.get(thread_id)
    elog("threads.get after resume", resumed)
    assert resumed["status"] == "idle"
    assert resumed["interrupts"] == {}
    assert resumed["values"]["foo"].endswith(" resumed")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_update_state_refreshes_thread_values() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    await client.runs.wait(
        thread_id,
        assistant["assistant_id"],
        input=_stress_input(),
    )

    await client.threads.update_state(thread_id, {"step_count": 42}, as_node="respond")

    fetched = await client.threads.get(thread_id)
    elog("threads.get after update_state", fetched)
    assert fetched["values"]["step_count"] == 42
