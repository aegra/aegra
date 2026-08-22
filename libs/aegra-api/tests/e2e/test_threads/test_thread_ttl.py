"""E2E tests for thread TTL: per-thread opt-in and POST /threads/prune.

Deterministic via /threads/prune instead of waiting on the background sweep
(its loop is unit-tested; the minimum useful interval is minutes). Uses the
hermetic stress_test graph — no LLM key required.
"""

import asyncio
import json

import httpx
import pytest
from langgraph_sdk.client import LangGraphClient

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client
from tests.e2e.test_threads.test_thread_deletion import _CHECKPOINT_COUNT_QUERIES, _count_checkpoint_rows

_RUN_INPUT = {"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 2})}]}


async def _run_to_completion(client: LangGraphClient, thread_id: str, assistant_id: str) -> None:
    run = await client.runs.create(thread_id=thread_id, assistant_id=assistant_id, input=_RUN_INPUT)
    await client.runs.join(thread_id, run["run_id"])
    finished = await client.runs.get(thread_id, run["run_id"])
    assert finished["status"] == "success", f"run ended as {finished['status']}"


def _prune() -> dict:
    resp = httpx.post(f"{settings.app.SERVER_URL}/threads/prune", timeout=30.0)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_prune_deletes_expired_thread() -> None:
    """An expired delete-strategy thread is fully removed, checkpoints included."""
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")

    # SDK ttl kwarg: number = minutes with strategy=delete. 0.001 min = 60 ms.
    thread = await client.threads.create(ttl=0.001)
    thread_id = thread["thread_id"]
    await _run_to_completion(client, thread_id, assistant["assistant_id"])

    before = await _count_checkpoint_rows(thread_id)
    elog("Checkpoint rows before prune", {"thread_id": thread_id, **before})
    assert before["checkpoints"] > 0

    await asyncio.sleep(1.0)  # let the 60 ms TTL expire
    result = _prune()
    elog("Prune result", result)
    assert result["deleted"] >= 1

    with pytest.raises(Exception, match="404|not found|Not Found"):
        await client.threads.get(thread_id)

    after = await _count_checkpoint_rows(thread_id)
    elog("Checkpoint rows after prune", {"thread_id": thread_id, **after})
    assert after == dict.fromkeys(_CHECKPOINT_COUNT_QUERIES, 0)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_prune_keep_latest_preserves_latest_state() -> None:
    """keep_latest compacts history to one checkpoint but keeps the thread usable."""
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")

    thread = await client.threads.create(ttl={"ttl": 0.001, "strategy": "keep_latest"})
    thread_id = thread["thread_id"]
    await _run_to_completion(client, thread_id, assistant["assistant_id"])

    history_before = await client.threads.get_history(thread_id)
    state_before = await client.threads.get_state(thread_id)
    assert len(history_before) > 1, "run should have produced multiple checkpoints"

    await asyncio.sleep(1.0)
    result = _prune()
    elog("Prune result", result)
    assert result["pruned"] >= 1

    # Thread survives with exactly the latest state
    thread_after = await client.threads.get(thread_id)
    assert thread_after["thread_id"] == thread_id
    history_after = await client.threads.get_history(thread_id)
    assert len(history_after) == 1
    state_after = await client.threads.get_state(thread_id)
    assert state_after["values"] == state_before["values"]

    # Strongest proof the kept checkpoint/blobs are self-consistent: the graph
    # resumes from the compacted state and completes again.
    await _run_to_completion(client, thread_id, assistant["assistant_id"])
    counts = await _count_checkpoint_rows(thread_id)
    elog("Checkpoint rows after second run", {"thread_id": thread_id, **counts})
    assert counts["checkpoints"] > 1

    await client.threads.delete(thread_id)
