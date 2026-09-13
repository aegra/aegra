"""Unit tests for run_preparation helpers."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from aegra_api.services import run_preparation as mod
from aegra_api.services.graph_factory import _FACTORY_CONTEXT_TYPES
from aegra_api.services.run_preparation import _validate_resume_command


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


class _Ctx(BaseModel):
    """Context type a graph factory declares via ``ServerRuntime[_Ctx]``."""

    prompt_version: int


class _StrictCtx(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: int


@pytest.fixture
def _declared_contexts() -> Iterator[dict[str, type | None]]:
    """Register factory context types for the duration of a test."""
    _FACTORY_CONTEXT_TYPES.clear()
    try:
        yield _FACTORY_CONTEXT_TYPES
    finally:
        _FACTORY_CONTEXT_TYPES.clear()


class TestValidateContextAgainstGraph:
    def test_graph_without_declared_type_accepts_anything(self, _declared_contexts: dict) -> None:
        mod._validate_context_against_graph({"anything": "goes"}, "untyped-graph")

    def test_plain_server_runtime_accepts_anything(self, _declared_contexts: dict) -> None:
        _declared_contexts["g"] = None
        mod._validate_context_against_graph({"anything": "goes"}, "g")

    def test_valid_context_passes(self, _declared_contexts: dict) -> None:
        _declared_contexts["g"] = _Ctx
        mod._validate_context_against_graph({"prompt_version": 3}, "g")

    def test_invalid_value_is_422_naming_the_field(self, _declared_contexts: dict) -> None:
        _declared_contexts["g"] = _Ctx

        with pytest.raises(HTTPException) as exc:
            mod._validate_context_against_graph({"prompt_version": "abc"}, "g")

        assert exc.value.status_code == 422
        assert "context.prompt_version" in exc.value.detail
        assert "g" in exc.value.detail

    def test_structured_errors_ride_along_in_details(self, _declared_contexts: dict) -> None:
        _declared_contexts["g"] = _Ctx

        with pytest.raises(HTTPException) as exc:
            mod._validate_context_against_graph({"prompt_version": "abc"}, "g")

        assert exc.value.details == {  # type: ignore[attr-defined]
            "errors": [
                {
                    "loc": ["context", "prompt_version"],
                    "msg": "Input should be a valid integer, unable to parse string as an integer",
                    "type": "int_parsing",
                }
            ]
        }

    def test_extra_forbid_rejects_instead_of_dropping_the_context(self, _declared_contexts: dict) -> None:
        """The point of ``extra="forbid"``: an unexpected key is an error, not a downgrade."""
        _declared_contexts["g"] = _StrictCtx

        with pytest.raises(HTTPException) as exc:
            mod._validate_context_against_graph({"prompt_version": 3, "typo": 1}, "g")

        assert exc.value.status_code == 422
        assert "context.typo" in exc.value.detail


class TestEveryRunCreationPathIsValidated:
    """Run rows are built in exactly one place, so one check covers every path.

    ``/threads/{id}/runs``, ``/threads/{id}/runs/stream``, ``/threads/{id}/runs/wait``,
    the stateless ``/runs*`` endpoints, the v2 ``run.start`` command, cron creation
    and the cron scheduler all reach the database through ``_prepare_run``. A new
    endpoint that builds its own ``RunORM`` would skip context validation, so fail
    here instead.
    """

    def test_run_rows_are_only_constructed_in_run_preparation(self) -> None:
        src = Path(mod.__file__).parent.parent
        constructing = sorted(
            path.relative_to(src).as_posix() for path in src.rglob("*.py") if "RunORM(" in path.read_text()
        )
        assert constructing == ["services/run_preparation.py"]
