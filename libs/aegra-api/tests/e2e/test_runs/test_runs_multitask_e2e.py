"""E2E tests for double-texting (multitask) strategies against a real server.

Uses the LLM-free ``stress_test`` graph, so no API tokens are needed (make e2e-dev / e2e-prod).
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings

GRAPH = "stress_test"
SLOW = '{"delay": 1.5, "steps": 3}'  # ~4.5s, valid JSON config for stress_test


def _base_url() -> str:
    return settings.app.SERVER_URL


async def _wait_terminal(client: httpx.AsyncClient, thread_id: str, run_id: str, timeout: float = 30.0) -> str:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"/threads/{thread_id}/runs/{run_id}")
        if resp.status_code == 404:
            return "deleted"
        resp.raise_for_status()
        status = resp.json().get("status")
        if status in ("success", "error", "interrupted"):
            return status
        await asyncio.sleep(0.5)
    raise AssertionError(f"run {run_id} did not settle within {timeout}s")


async def _setup(client: httpx.AsyncClient) -> tuple[str, str]:
    a = await client.post("/assistants", json={"graph_id": GRAPH, "if_exists": "do_nothing"})
    a.raise_for_status()
    t = await client.post("/threads", json={})
    t.raise_for_status()
    return a.json()["assistant_id"], t.json()["thread_id"]


def _body(assistant_id: str, strategy: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"assistant_id": assistant_id, "input": {"messages": [{"role": "user", "content": SLOW}]}}
    if strategy is not None:
        body["multitask_strategy"] = strategy
    return body


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_default_strategy_enqueues_e2e() -> None:
    """A run created with no strategy queues behind the active run (default=enqueue)."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_body(aid, "enqueue"))
        a.raise_for_status()
        await asyncio.sleep(1.0)
        b = await client.post(f"/threads/{tid}/runs", json=_body(aid, None))
        b.raise_for_status()

        # Internal 'queued' is reported as 'pending' on the wire (SDK vocabulary).
        assert b.json()["status"] == "pending"
        assert await _wait_terminal(client, tid, a.json()["run_id"]) == "success"
        assert await _wait_terminal(client, tid, b.json()["run_id"]) == "success"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_reject_returns_409_e2e() -> None:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_body(aid, "enqueue"))
        a.raise_for_status()
        await asyncio.sleep(1.0)

        b = await client.post(f"/threads/{tid}/runs", json=_body(aid, "reject"))

        assert b.status_code == 409
        await _wait_terminal(client, tid, a.json()["run_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_enqueue_runs_after_active_e2e() -> None:
    """The queued run only leaves 'queued' after the active run finishes."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_body(aid, "enqueue"))
        a.raise_for_status()
        rid_a = a.json()["run_id"]
        await asyncio.sleep(1.0)
        b = await client.post(f"/threads/{tid}/runs", json=_body(aid, "enqueue"))
        b.raise_for_status()
        rid_b = b.json()["run_id"]

        # While A runs, B must still be parked (not executing concurrently);
        # internal 'queued' surfaces as 'pending' with no output yet.
        mid = await client.get(f"/threads/{tid}/runs/{rid_b}")
        assert mid.json()["status"] == "pending"
        assert mid.json()["output"] is None

        assert await _wait_terminal(client, tid, rid_a) == "success"
        assert await _wait_terminal(client, tid, rid_b) == "success"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_interrupt_drops_queued_double_text_e2e() -> None:
    """interrupt over [running, queued] drops the queued double-text — it does not run afterward."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_slow_marker_body(aid, "ACTIVE-A"))
        a.raise_for_status()
        rid_a = a.json()["run_id"]
        await asyncio.sleep(1.0)  # A is running
        b = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "QUEUED-B", "enqueue"))
        b.raise_for_status()
        rid_b = b.json()["run_id"]
        assert b.json()["status"] == "pending"  # internal 'queued'

        # Interrupt supersedes both the running A and the queued B, then runs C now.
        c = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "NEW-C", "interrupt"))
        c.raise_for_status()

        assert await _wait_terminal(client, tid, rid_a) == "interrupted"
        assert await _wait_terminal(client, tid, rid_b) == "interrupted"  # dropped, not run later
        assert await _wait_terminal(client, tid, c.json()["run_id"]) == "success"
        raw = str((await client.get(f"/threads/{tid}/state")).json().get("values", {}).get("messages", []))
        assert "QUEUED-B" not in raw  # the stale double-text never executed


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_interrupt_cancels_active_e2e() -> None:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_body(aid, "enqueue"))
        a.raise_for_status()
        rid_a = a.json()["run_id"]
        await asyncio.sleep(1.0)
        b = await client.post(f"/threads/{tid}/runs", json=_body(aid, "interrupt"))
        b.raise_for_status()

        assert await _wait_terminal(client, tid, rid_a) == "interrupted"
        assert await _wait_terminal(client, tid, b.json()["run_id"]) == "success"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_rollback_cancels_active_e2e() -> None:
    """rollback over an active run cancels it (kept as interrupted, not deleted)."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_body(aid, "enqueue"))
        a.raise_for_status()
        rid_a = a.json()["run_id"]
        await asyncio.sleep(1.0)
        b = await client.post(f"/threads/{tid}/runs", json=_body(aid, "rollback"))
        b.raise_for_status()

        assert await _wait_terminal(client, tid, rid_a) == "interrupted"  # row kept, not 404
        assert await _wait_terminal(client, tid, b.json()["run_id"]) == "success"


def _marker_body(assistant_id: str, marker: str, strategy: str | None = None) -> dict[str, Any]:
    """Valid stress_test JSON config carrying a unique marker that survives in the echo."""
    content = json.dumps({"delay": 0.2, "steps": 1, "_m": marker})
    body: dict[str, Any] = {"assistant_id": assistant_id, "input": {"messages": [{"role": "user", "content": content}]}}
    if strategy is not None:
        body["multitask_strategy"] = strategy
    return body


def _interrupt_marker_body(
    assistant_id: str, marker: str, strategy: str | None = None, delay: float = 0.1
) -> dict[str, Any]:
    """stress_test config that pauses on a human-in-the-loop interrupt() (LLM-free HITL)."""
    content = json.dumps({"delay": delay, "steps": 1, "interrupt": True, "_m": marker})
    body: dict[str, Any] = {"assistant_id": assistant_id, "input": {"messages": [{"role": "user", "content": content}]}}
    if strategy is not None:
        body["multitask_strategy"] = strategy
    return body


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_rollback_idle_runs_fresh_and_reverts_nothing_e2e() -> None:
    """On an idle thread rollback only adds a turn: like LangGraph Platform it reverts in-flight runs only."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)

        p = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "PRIOR"))
        p.raise_for_status()
        assert await _wait_terminal(client, tid, p.json()["run_id"]) == "success"
        a = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "DONE"))
        a.raise_for_status()
        assert await _wait_terminal(client, tid, a.json()["run_id"]) == "success"

        b = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "NEW", "rollback"))
        b.raise_for_status()
        assert await _wait_terminal(client, tid, b.json()["run_id"]) == "success"

        raw = str((await client.get(f"/threads/{tid}/state")).json().get("values", {}).get("messages", []))
        assert "PRIOR" in raw  # prior state preserved
        assert "NEW" in raw  # new run ran
        assert "DONE" in raw  # the completed run is NOT reverted (no undo-last-turn footgun)


def _slow_marker_body(assistant_id: str, marker: str, strategy: str | None = None) -> dict[str, Any]:
    content = json.dumps({"delay": 2, "steps": 3, "_m": marker})  # ~6s, stays active long enough to preempt
    body: dict[str, Any] = {"assistant_id": assistant_id, "input": {"messages": [{"role": "user", "content": content}]}}
    if strategy is not None:
        body["multitask_strategy"] = strategy
    return body


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_rollback_reverts_active_run_e2e() -> None:
    """rollback over an ACTIVE run forks from before it, reverting its writes."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        p = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "PRIOR"))
        p.raise_for_status()
        assert await _wait_terminal(client, tid, p.json()["run_id"]) == "success"

        a = await client.post(f"/threads/{tid}/runs", json=_slow_marker_body(aid, "ACTIVE-A"))
        a.raise_for_status()
        await asyncio.sleep(1.0)  # A is running
        b = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "NEW", "rollback"))
        b.raise_for_status()

        assert await _wait_terminal(client, tid, a.json()["run_id"]) == "interrupted"
        assert await _wait_terminal(client, tid, b.json()["run_id"]) == "success"
        raw = str((await client.get(f"/threads/{tid}/state")).json().get("values", {}).get("messages", []))
        assert "PRIOR" in raw
        assert "NEW" in raw
        assert "ACTIVE-A" not in raw  # the cancelled run's writes are reverted


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fresh_run_on_hitl_pause_admitted_by_default_e2e() -> None:
    """Default policy (admit, as LangGraph Platform): fresh input on a paused thread runs and clears the pause."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)

        # A run that pauses on interrupt() ends 'interrupted' with a pending interrupt.
        paused = await client.post(f"/threads/{tid}/runs", json=_interrupt_marker_body(aid, "PAUSED"))
        paused.raise_for_status()
        assert await _wait_terminal(client, tid, paused.json()["run_id"]) == "interrupted"
        assert (await client.get(f"/threads/{tid}/state")).json().get("interrupts")  # pending interrupt present

        # Fresh input is admitted: the graph restarts from __start__ and the pending interrupt is dropped.
        b = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "NEW", "rollback"))
        b.raise_for_status()
        assert await _wait_terminal(client, tid, b.json()["run_id"]) == "success"

        state = (await client.get(f"/threads/{tid}/state")).json()
        assert not state.get("interrupts")  # the pause was consumed, not preserved
        assert "NEW" in str(state.get("values", {}).get("messages", []))


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_queued_run_promoted_onto_hitl_pause_by_default_e2e() -> None:
    """Default policy: a run enqueued before the thread paused is promoted onto the pause once it frees."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)

        # A runs a busy window, then pauses on interrupt(); B enqueues while A is still running.
        a = await client.post(f"/threads/{tid}/runs", json=_interrupt_marker_body(aid, "PAUSED", delay=2.0))
        a.raise_for_status()
        await asyncio.sleep(0.8)  # A is running (busy), not yet paused
        b = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "QUEUED-B", "enqueue"))
        b.raise_for_status()
        rid_b = b.json()["run_id"]
        assert b.json()["status"] == "pending"  # internal 'queued'

        assert await _wait_terminal(client, tid, a.json()["run_id"]) == "interrupted"
        # A's finalize promotes B onto the paused thread; B's fresh input discards the pause.
        assert await _wait_terminal(client, tid, rid_b) == "success"
        state = (await client.get(f"/threads/{tid}/state")).json()
        assert not state.get("interrupts")
        assert "QUEUED-B" in str(state.get("values", {}).get("messages", []))


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_reject_creates_exactly_one_409_e2e() -> None:
    """Two truly concurrent creates with strategy=reject: the FOR UPDATE admission
    gate must serialize them so exactly one wins and exactly one gets 409."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        # Warm the thread row so both racers hit the same locked row.
        warm = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "WARM"))
        warm.raise_for_status()
        assert await _wait_terminal(client, tid, warm.json()["run_id"]) == "success"

        results = await asyncio.gather(
            client.post(f"/threads/{tid}/runs", json=_body(aid, "reject")),
            client.post(f"/threads/{tid}/runs", json=_body(aid, "reject")),
        )
        codes = sorted(r.status_code for r in results)

        assert codes == [200, 409], f"gate must admit exactly one racer, got {codes}"
        winner = next(r for r in results if r.status_code == 200)
        assert await _wait_terminal(client, tid, winner.json()["run_id"]) == "success"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_two_queued_runs_promote_fifo_e2e() -> None:
    """Two runs enqueued behind an active one must execute in creation order (FIFO)."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_slow_marker_body(aid, "HEAD"))
        a.raise_for_status()
        await asyncio.sleep(1.0)  # HEAD is running
        b = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "FIRST", "enqueue"))
        b.raise_for_status()
        await asyncio.sleep(0.3)  # order the two queued runs by created_at
        c = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "SECOND", "enqueue"))
        c.raise_for_status()

        assert await _wait_terminal(client, tid, a.json()["run_id"]) == "success"
        assert await _wait_terminal(client, tid, b.json()["run_id"]) == "success"
        assert await _wait_terminal(client, tid, c.json()["run_id"]) == "success"

        # FIFO: the echo transcript must show FIRST before SECOND.
        raw = str((await client.get(f"/threads/{tid}/state")).json().get("values", {}).get("messages", []))
        assert "FIRST" in raw and "SECOND" in raw
        assert raw.index("FIRST") < raw.index("SECOND"), "queued runs must promote oldest-first"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancel_endpoint_resets_thread_and_promotes_queue_e2e() -> None:
    """POST .../cancel on the active run must not strand the thread 'busy' —
    the queued double-text behind it is promoted and the thread settles."""
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60) as client:
        aid, tid = await _setup(client)
        a = await client.post(f"/threads/{tid}/runs", json=_slow_marker_body(aid, "CANCELLED"))
        a.raise_for_status()
        rid_a = a.json()["run_id"]
        await asyncio.sleep(1.0)  # A is running
        b = await client.post(f"/threads/{tid}/runs", json=_marker_body(aid, "QUEUED", "enqueue"))
        b.raise_for_status()

        (await client.post(f"/threads/{tid}/runs/{rid_a}/cancel")).raise_for_status()

        assert await _wait_terminal(client, tid, rid_a) == "interrupted"
        # The queued run is promoted promptly (no reliance on the 15-30s sweeps).
        assert await _wait_terminal(client, tid, b.json()["run_id"], timeout=20.0) == "success"
        thread = await client.get(f"/threads/{tid}")
        assert thread.json()["status"] == "idle"  # not stranded 'busy'
