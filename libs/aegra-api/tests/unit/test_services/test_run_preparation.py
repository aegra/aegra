"""Unit tests for run_preparation helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from aegra_api.services import run_preparation as mod
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


class TestPrepareRunAssistantIdentity:
    """The resolved assistant becomes the run's server-authoritative identity."""

    @staticmethod
    def _session(assistant: object, thread: object) -> AsyncMock:
        """A session that answers the assistant lookup, then the thread lookup."""
        session = AsyncMock()
        session.scalar = AsyncMock(side_effect=[assistant, thread])
        session.execute = AsyncMock(return_value=MagicMock(rowcount=1))
        session.add = MagicMock()
        return session

    @staticmethod
    def _assistant() -> SimpleNamespace:
        return SimpleNamespace(
            assistant_id="asst-1",
            graph_id="graph-1",
            config={},
            context={},
        )

    async def _prepare(self, monkeypatch: pytest.MonkeyPatch, request_config: dict | None) -> object:
        from aegra_api.models import RunCreate, User

        thread = SimpleNamespace(
            thread_id="thread-1",
            status="idle",
            metadata_json={"owner": "user-1"},
            user_id="user-1",
        )
        session = self._session(self._assistant(), thread)

        service = MagicMock()
        service.list_graphs = MagicMock(return_value={"graph-1": "graph.py:graph"})
        monkeypatch.setattr(mod, "get_langgraph_service", lambda: service)

        submitted: list[object] = []
        submit = AsyncMock(side_effect=lambda job: submitted.append(job))
        monkeypatch.setattr(mod.executor, "submit", submit)

        await mod._prepare_run(
            session,
            "thread-1",
            RunCreate(
                assistant_id="asst-1",
                input={"message": "hi"},
                config=request_config,
            ),
            User(identity="user-1"),
            initial_status="pending",
        )

        return submitted[0]

    async def test_identity_carries_the_resolved_assistant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        job = await self._prepare(monkeypatch, None)

        assert job.identity.assistant_id == "asst-1"

    async def test_client_supplied_assistant_id_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Posting another assistant's id under config.configurable must not take effect.

        The graph is built and configured from this value, so honoring the
        client's would let a caller reach another tenant's configuration.
        """
        from aegra_api.services.run_executor import _build_run_config

        job = await self._prepare(
            monkeypatch,
            {"configurable": {"assistant_id": "victim-assistant"}},
        )

        assert job.identity.assistant_id == "asst-1"
        assert _build_run_config(job)["configurable"]["assistant_id"] == "asst-1"
