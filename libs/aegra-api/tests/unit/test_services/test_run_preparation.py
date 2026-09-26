"""Unit tests for run_preparation helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from aegra_api.models.runs import RunCreate
from aegra_api.services import run_preparation as mod
from aegra_api.services.run_preparation import _resolve_checkpoint, _validate_resume_command


@pytest.fixture(autouse=True)
def _fast_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the resume-settle backoff so reject paths don't wait."""
    monkeypatch.setattr(mod, "_RESUME_SETTLE_INTERVAL_SECONDS", 0)


def _thread(status: str) -> SimpleNamespace:
    return SimpleNamespace(status=status)


def _session_returning(thread: object) -> AsyncMock:
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=thread)
    return session


def _patch_fresh_sessions(monkeypatch: pytest.MonkeyPatch, *threads: object) -> None:
    """Make run_preparation's fresh-session poll yield the given threads in order."""
    seq = list(threads)

    async def scalar(_stmt: object) -> object:
        return seq.pop(0) if len(seq) > 1 else (seq[0] if seq else None)

    fresh = AsyncMock()
    fresh.scalar = AsyncMock(side_effect=scalar)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=fresh)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_session_maker", lambda: MagicMock(return_value=ctx))


class TestValidateResumeCommand:
    async def test_resume_on_interrupted_thread_passes(self) -> None:
        session = _session_returning(_thread("interrupted"))
        await _validate_resume_command(session, "t1", {"resume": "yes"})

    async def test_resume_none_on_non_interrupted_thread_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """{"resume": None} must still be guarded — None is a valid resume payload."""
        _patch_fresh_sessions(monkeypatch, _thread("idle"))
        session = _session_returning(_thread("idle"))
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 400

    async def test_resume_settles_when_status_flips_to_interrupted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The interrupt reaches the client before finalize commits 'interrupted';
        the guard polls a fresh session and accepts once the status settles."""
        _patch_fresh_sessions(monkeypatch, _thread("busy"), _thread("interrupted"))
        session = _session_returning(_thread("busy"))  # first (request-session) read is stale
        await _validate_resume_command(session, "t1", {"resume": "yes"})

    async def test_resume_on_missing_thread_is_404(self) -> None:
        session = _session_returning(None)
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 404

    async def test_non_resume_command_skips_check(self) -> None:
        session = _session_returning(_thread("idle"))
        await _validate_resume_command(session, "t1", {"goto": "node"})
        session.scalar.assert_not_awaited()

    async def test_none_command_skips_check(self) -> None:
        session = _session_returning(_thread("idle"))
        await _validate_resume_command(session, "t1", None)
        session.scalar.assert_not_awaited()


_CHECKPOINT_ID = "1ef4f797-8335-6428-8001-8a1503f9b875"
_OTHER_CHECKPOINT_ID = "1ef4f797-8335-6428-8001-8a1503f9b876"


class TestResolveCheckpoint:
    def test_returns_checkpoint_unchanged_without_checkpoint_id(self) -> None:
        request = RunCreate(assistant_id="agent", checkpoint={"checkpoint_id": "chk-1", "checkpoint_ns": ""})
        assert _resolve_checkpoint(request) == {"checkpoint_id": "chk-1", "checkpoint_ns": ""}

    def test_returns_none_when_neither_is_set(self) -> None:
        assert _resolve_checkpoint(RunCreate(assistant_id="agent", input={"x": 1})) is None

    def test_builds_checkpoint_from_top_level_checkpoint_id(self) -> None:
        request = RunCreate(assistant_id="agent", checkpoint_id=_CHECKPOINT_ID)
        assert _resolve_checkpoint(request) == {"checkpoint_id": _CHECKPOINT_ID}

    def test_keeps_other_checkpoint_keys(self) -> None:
        request = RunCreate(assistant_id="agent", checkpoint_id=_CHECKPOINT_ID, checkpoint={"checkpoint_ns": "sub"})
        assert _resolve_checkpoint(request) == {"checkpoint_id": _CHECKPOINT_ID, "checkpoint_ns": "sub"}

    def test_checkpoint_dict_wins_over_top_level_checkpoint_id(self) -> None:
        request = RunCreate(
            assistant_id="agent",
            checkpoint_id=_CHECKPOINT_ID,
            checkpoint={"checkpoint_id": _OTHER_CHECKPOINT_ID},
        )
        assert _resolve_checkpoint(request) == {"checkpoint_id": _OTHER_CHECKPOINT_ID}
