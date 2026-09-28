"""E2E: the SDK's top-level ``checkpoint_id`` run param replays from that checkpoint."""

import json

import pytest

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_with_only_checkpoint_id_replays_from_that_checkpoint() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    assistant_id = assistant["assistant_id"]
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    await client.runs.wait(
        thread_id,
        assistant_id,
        input={"messages": [{"role": "user", "content": json.dumps({"steps": 1})}]},
    )
    history = await client.threads.get_history(thread_id)
    fork_target = next(state for state in history if state["next"])
    checkpoint_id = fork_target["checkpoint"]["checkpoint_id"]
    elog("fork target", {"checkpoint_id": checkpoint_id, "next": fork_target["next"]})

    run = await client.runs.create(thread_id, assistant_id, checkpoint_id=checkpoint_id)
    await client.runs.join(thread_id, run["run_id"])

    finished = await client.runs.get(thread_id, run["run_id"])
    assert finished["status"] == "success"
    forked_history = await client.threads.get_history(thread_id)
    forked = [state for state in forked_history if state["checkpoint"]["checkpoint_id"] != checkpoint_id]
    assert len(forked_history) > len(history)
    assert any(
        (state.get("parent_checkpoint") or {}).get("checkpoint_id") == checkpoint_id and state not in history
        for state in forked
    )
