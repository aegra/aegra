import uuid

import pytest

from tests.e2e._utils import check_and_skip_if_geo_blocked, elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_copy_returns_new_id_and_preserves_status() -> None:
    """Copy of an empty thread returns a new ``thread_id`` and inherits status.

    Empty-thread baseline: no checkpoint history, just the thread row, so the
    test isolates the row-copy semantics from the checkpoint-chain semantics.
    """
    client = get_e2e_client()

    src = await client.threads.create(metadata={"label": "copy-empty-source"})
    src_id = src["thread_id"]
    src_status = src["status"]
    elog("Created source thread", {"thread_id": src_id, "status": src_status})

    new_thread = await client.threads.copy(src_id)
    elog("Copied thread", new_thread)

    assert new_thread["thread_id"] != src_id, "copy must mint a fresh thread_id"
    assert new_thread["status"] == src_status, "status inherited from source"

    fetched = await client.threads.get(new_thread["thread_id"])
    assert fetched["thread_id"] == new_thread["thread_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_copy_preserves_checkpoint_chain() -> None:
    """A run produces checkpoints; ``copy`` keeps the same ``checkpoint_id``.

    Verifies the SQL-level INSERT...SELECT semantics end-to-end: after a run,
    the source has a checkpoint chain; the copy must expose that chain under
    its own ``thread_id`` with identical ``checkpoint_id`` values.
    """
    client = get_e2e_client()

    src = await client.threads.create()
    src_id = src["thread_id"]

    await client.runs.wait(
        thread_id=src_id,
        assistant_id="agent",
        input={"messages": [{"role": "human", "content": "Reply with the single word 'ok'."}]},
    )
    runs_list = await client.runs.list(src_id)
    assert runs_list, "Expected source run to be recorded"
    check_and_skip_if_geo_blocked(runs_list[0])
    assert runs_list[0]["status"] in ("success", "interrupted"), (
        f"source run failed unexpectedly: {runs_list[0]['status']}"
    )

    src_state = await client.threads.get_state(thread_id=src_id)
    src_checkpoint_id = src_state["checkpoint"]["checkpoint_id"]
    src_history = await client.threads.get_history(src_id)
    elog(
        "Source run + state",
        {
            "checkpoint_id": src_checkpoint_id,
            "history_len": len(src_history),
            "run_status": runs_list[0]["status"],
        },
    )
    assert src_checkpoint_id is not None
    assert src_history, "Expected non-empty checkpoint history on source"

    new_thread = await client.threads.copy(src_id)
    new_id = new_thread["thread_id"]

    new_state = await client.threads.get_state(thread_id=new_id)
    new_checkpoint_id = new_state["checkpoint"]["checkpoint_id"]
    new_history = await client.threads.get_history(new_id)
    elog(
        "Copy state",
        {
            "thread_id": new_id,
            "checkpoint_id": new_checkpoint_id,
            "history_len": len(new_history),
        },
    )

    assert new_checkpoint_id == src_checkpoint_id, "checkpoint_id must be identical between source and copy"
    assert len(new_history) == len(src_history), "checkpoint history length must match between source and copy"
    src_chain = [c["checkpoint"]["checkpoint_id"] for c in src_history]
    new_chain = [c["checkpoint"]["checkpoint_id"] for c in new_history]
    assert src_chain == new_chain, "checkpoint chain must be identical"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_copy_404_for_missing_source() -> None:
    """Copying a thread the caller does not own (or that doesn't exist) returns 404.

    Uses a freshly-minted UUID as source; ownership check + non-existence
    converge on the same 404 response.
    """
    client = get_e2e_client()
    missing_id = str(uuid.uuid4())

    with pytest.raises(Exception) as excinfo:
        await client.threads.copy(missing_id)

    msg = str(excinfo.value)
    assert "404" in msg or "not found" in msg.lower(), f"expected 404 / not-found error, got: {msg}"
    elog("Copy of missing thread surfaced expected error", {"error": msg})
