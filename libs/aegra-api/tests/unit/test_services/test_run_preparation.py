"""Unit tests for run_preparation helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from aegra_api.services.run_preparation import _validate_resume_command


def _session_with_thread(status: str | None) -> AsyncMock:
    """A session whose scalar() returns a thread of the given status, or None."""
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=SimpleNamespace(status=status) if status else None)
    return session


class TestValidateResumeCommand:
    async def test_resume_on_interrupted_thread_passes(self) -> None:
        session = _session_with_thread("interrupted")
        await _validate_resume_command(session, "t1", {"resume": "yes"})

    async def test_resume_none_on_non_interrupted_thread_is_rejected(self) -> None:
        """{"resume": None} must still be guarded — None is a valid resume payload."""
        session = _session_with_thread("idle")
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 400

    async def test_resume_value_on_busy_thread_is_rejected(self) -> None:
        session = _session_with_thread("busy")
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": "yes"})
        assert exc.value.status_code == 400

    async def test_resume_on_missing_thread_is_404(self) -> None:
        session = _session_with_thread(None)
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 404

    async def test_non_resume_command_skips_check(self) -> None:
        session = _session_with_thread("idle")
        await _validate_resume_command(session, "t1", {"goto": "node"})
        session.scalar.assert_not_awaited()

    async def test_none_command_skips_check(self) -> None:
        session = _session_with_thread("idle")
        await _validate_resume_command(session, "t1", None)
        session.scalar.assert_not_awaited()
