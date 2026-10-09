"""E2E: GET /threads/{thread_id} includes documented Thread state fields."""

from typing import Any

import pytest

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("graph_id", [None, "stress_test"])
async def test_get_thread_includes_empty_state_fields_without_checkpoint(graph_id: str | None) -> None:
    """A thread with no runs still returns the documented state keys."""
    client = get_e2e_client()
    created: dict[str, Any] = await client.threads.create(metadata={"graph_id": graph_id} if graph_id else {})
    thread_id = created["thread_id"]

    fetched: dict[str, Any] = await client.threads.get(thread_id)
    elog("GET thread (no checkpoint)", fetched)

    assert fetched["thread_id"] == thread_id
    assert fetched["values"] == {}
    assert fetched["interrupts"] == {}
    assert fetched["config"] == {}
    assert fetched.get("state_updated_at") is None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_thread_includes_latest_checkpoint_values_after_run() -> None:
    """After a run, GET /threads/{id} exposes the latest checkpoint values."""
    client = get_e2e_client()
    created: dict[str, Any] = await client.threads.create()
    thread_id = created["thread_id"]

    await client.runs.wait(
        thread_id=thread_id,
        assistant_id="stress_test",
        input={"messages": [{"role": "user", "content": '{"delay": 0.1, "steps": 1}'}]},
    )

    fetched: dict[str, Any] = await client.threads.get(thread_id)
    elog("GET thread after run", fetched)

    state = await client.threads.get_state(thread_id)
    messages = fetched["values"]["messages"]
    assert messages[-1]["content"]
    assert fetched["values"] == state["values"]
    assert fetched["config"]["configurable"]["checkpoint_id"] == state["checkpoint"]["checkpoint_id"]
    assert fetched["state_updated_at"] == state["created_at"]
    assert isinstance(fetched["interrupts"], dict)
    assert fetched["state_updated_at"] is not None
    assert "next" not in fetched
    assert "checkpoint" not in fetched

    await client.threads.update_state(thread_id, values={"step_count": 7}, as_node="process")
    updated = await client.threads.get(thread_id)
    latest = await client.threads.get_state(thread_id)
    assert updated["values"]["step_count"] == 7
    assert updated["values"] == latest["values"]
    assert updated["config"]["configurable"]["checkpoint_id"] == latest["checkpoint"]["checkpoint_id"]
    assert updated["config"]["configurable"]["checkpoint_id"] != fetched["config"]["configurable"]["checkpoint_id"]
    assert updated["state_updated_at"] == latest["created_at"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_thread_returns_task_keyed_interrupts_and_clears_them_after_resume() -> None:
    client = get_e2e_client()
    created: dict[str, Any] = await client.threads.create()
    thread_id = created["thread_id"]
    await client.runs.wait(thread_id, "subgraph_hitl_agent", input={"foo": "start"})

    fetched = await client.threads.get(thread_id)
    state = await client.threads.get_state(thread_id)
    expected = {task["id"]: task["interrupts"] for task in state["tasks"] if task["interrupts"]}
    elog("GET interrupted thread", fetched)
    assert expected
    assert fetched["interrupts"] == expected
    assert fetched["config"]["configurable"]["checkpoint_id"] == state["checkpoint"]["checkpoint_id"]

    await client.runs.wait(thread_id, "subgraph_hitl_agent", command={"resume": "approved"})
    resumed = await client.threads.get(thread_id)
    state = await client.threads.get_state(thread_id)
    assert resumed["interrupts"] == {}
    assert resumed["values"] == state["values"]
    assert resumed["values"]["foo"] == "Initial subgraph value.approved"
