"""E2E: GET /threads/{thread_id} includes documented Thread state fields."""

from typing import Any

import pytest

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_thread_includes_empty_state_fields_without_checkpoint() -> None:
    """A thread with no runs still returns the documented state keys."""
    client = get_e2e_client()
    created: dict[str, Any] = await client.threads.create()
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

    messages = fetched["values"]["messages"]
    assert messages[-1]["content"]
    assert isinstance(fetched["interrupts"], dict)
    assert fetched["state_updated_at"] is not None
    assert "next" not in fetched
    assert "checkpoint" not in fetched
