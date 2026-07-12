"""Tests for per-assistant secret handling (encryption + execution resolution)."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet

from aegra_api.core import crypto
from aegra_api.models import Assistant
from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.services import run_executor


def test_assistant_response_has_no_secrets_field() -> None:
    """A secret must never be serializable in an assistant response."""
    assert "secrets" not in Assistant.model_fields


@pytest.fixture
def enc_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AEGRA_SECRET_KEY", key)
    crypto._fernet.cache_clear()
    yield key
    crypto._fernet.cache_clear()


def _mock_session_maker(monkeypatch: pytest.MonkeyPatch, secrets_value: dict | None) -> None:
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=secrets_value)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(run_executor, "get_session_maker", lambda: MagicMock(return_value=cm))


class TestResolveContext:
    """run_executor._resolve_context decrypts assistant secrets into the runtime context."""

    @pytest.mark.asyncio
    async def test_merges_decrypted_secrets(self, monkeypatch: pytest.MonkeyPatch, enc_key: str) -> None:
        token = crypto.encrypt("sk-real")
        _mock_session_maker(monkeypatch, {"api_key": token})
        job = RunJob(
            identity=RunIdentity(run_id="r", thread_id="t", graph_id="g", assistant_id="a1"),
            user=User(user_id="u"),
            execution=RunExecution(context={"model": "gpt"}),
        )
        ctx = await run_executor._resolve_context(job)
        assert ctx["api_key"] == "sk-real"
        assert ctx["model"] == "gpt"

    @pytest.mark.asyncio
    async def test_no_assistant_id_returns_base_context(self) -> None:
        job = RunJob(
            identity=RunIdentity(run_id="r", thread_id="t", graph_id="g"),
            user=User(user_id="u"),
            execution=RunExecution(context={"model": "gpt"}),
        )
        ctx = await run_executor._resolve_context(job)
        assert ctx == {"model": "gpt"}
