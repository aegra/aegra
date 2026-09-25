"""E2E: the run's ``durability`` reaches LangGraph, so ``exit`` persists one checkpoint per run."""

import json
from typing import Any

import pytest

from tests.e2e._utils import elog, get_e2e_client

_STEPS = 2
_INPUT: dict[str, Any] = {"messages": [{"role": "user", "content": json.dumps({"delay": 0, "steps": _STEPS})}]}


async def _history_after_one_run(**run_kwargs: Any) -> list[dict[str, Any]]:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    await client.runs.wait(thread["thread_id"], assistant["assistant_id"], input=_INPUT, **run_kwargs)
    history = await client.threads.get_history(thread["thread_id"])
    elog("history", {"run_kwargs": run_kwargs, "checkpoints": len(history)})
    return history


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_default_durability_checkpoints_every_step() -> None:
    history = await _history_after_one_run()

    assert len(history) > 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_exit_durability_persists_only_the_final_state() -> None:
    history = await _history_after_one_run(durability="exit")

    assert len(history) == 1
    assert history[0]["values"]["step_count"] == _STEPS
    assert history[0]["next"] == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_checkpoint_during_false_behaves_like_exit() -> None:
    history = await _history_after_one_run(checkpoint_during=False)

    assert len(history) == 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sync_durability_checkpoints_every_step() -> None:
    history = await _history_after_one_run(durability="sync")

    assert len(history) > 1
    assert history[0]["values"]["step_count"] == _STEPS
