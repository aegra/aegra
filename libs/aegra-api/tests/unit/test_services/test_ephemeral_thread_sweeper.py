"""Tests for stale stateless-run thread reclamation."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from aegra_api.services import ephemeral_thread_sweeper as mod


def test_orphan_claim_requires_ephemeral_old_and_terminal_threads() -> None:
    """The claim query cannot select persistent, fresh, or active threads."""
    sql = str(
        mod._orphaned_threads_stmt(cutoff=datetime.now(UTC), limit=10).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )

    assert "thread.is_ephemeral IS true" in sql
    assert "thread.updated_at <=" in sql
    assert "NOT (EXISTS (SELECT" in sql
    assert "runs.status IN ('pending', 'running')" in sql
    assert "FOR UPDATE OF thread SKIP LOCKED" in sql


@pytest.mark.asyncio
async def test_sweep_deletes_checkpoints_before_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reclaimed row deletes LangGraph state before its metadata row."""
    thread = SimpleNamespace(thread_id="t1")
    session = AsyncMock()
    session.scalar.return_value = None
    result = MagicMock()
    result.all.return_value = [thread]
    empty = MagicMock()
    empty.all.return_value = []
    session.scalars.side_effect = [result, empty]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_session_maker", lambda: MagicMock(return_value=context))

    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock()
    monkeypatch.setattr(mod.db_manager, "get_checkpointer", lambda: checkpointer)

    claimed, deleted, errors = await mod.sweep_orphaned_threads()

    assert (claimed, deleted, errors) == (1, 1, 0)
    checkpointer.adelete_thread.assert_awaited_once_with("t1")
    session.delete.assert_awaited_once_with(thread)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_sweep_keeps_thread_when_checkpoint_delete_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint failures leave the row available for a later retry."""
    thread = SimpleNamespace(thread_id="t1")
    session = AsyncMock()
    session.scalar.return_value = None
    result = MagicMock()
    result.all.return_value = [thread]
    empty = MagicMock()
    empty.all.return_value = []
    session.scalars.side_effect = [result, empty]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_session_maker", lambda: MagicMock(return_value=context))

    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock(side_effect=OSError("checkpoint unavailable"))
    monkeypatch.setattr(mod.db_manager, "get_checkpointer", lambda: checkpointer)

    claimed, deleted, errors = await mod.sweep_orphaned_threads()

    assert (claimed, deleted, errors) == (1, 0, 1)
    session.delete.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_sweep_does_not_count_rows_when_commit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rolled-back metadata transaction is reported as an error, not a deletion."""
    thread = SimpleNamespace(thread_id="t1")
    session = AsyncMock()
    session.scalar.return_value = None
    result = MagicMock()
    result.all.return_value = [thread]
    empty = MagicMock()
    empty.all.return_value = []
    session.scalars.side_effect = [result, empty]
    session.commit.side_effect = OSError("database unavailable")
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_session_maker", lambda: MagicMock(return_value=context))

    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock()
    monkeypatch.setattr(mod.db_manager, "get_checkpointer", lambda: checkpointer)

    claimed, deleted, errors = await mod.sweep_orphaned_threads()

    assert (claimed, deleted, errors) == (1, 0, 1)
    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_sweep_skips_thread_with_active_run_after_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh active-run check protects a thread committed during lock wait."""
    thread = SimpleNamespace(thread_id="t1")
    session = AsyncMock()
    session.scalar.return_value = "active-run"
    result = MagicMock()
    result.all.return_value = [thread]
    empty = MagicMock()
    empty.all.return_value = []
    session.scalars.side_effect = [result, empty]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_session_maker", lambda: MagicMock(return_value=context))

    claimed, deleted, errors = await mod.sweep_orphaned_threads()

    assert (claimed, deleted, errors) == (1, 0, 0)
    session.delete.assert_not_awaited()
    session.commit.assert_awaited_once()
