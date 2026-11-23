"""``if_not_exists`` on run creation: 'reject' turns a missing thread into a 404 (#503)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from aegra_api.api.runs import create_and_stream_run, create_run, wait_for_run
from aegra_api.api.stateless_runs import stateless_create_run, stateless_stream_run, stateless_wait_for_run
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import RunCreate, User

_USER = User(identity="test-user", scopes=[])
_THREAD_ID = "thread-1"
_ENDPOINTS = ["create", "stream", "wait"]


class _ReachedPrepare(Exception):
    """Raised by the stubbed ``_prepare_run``: the handler got past the thread check."""


def _own_thread() -> ThreadORM:
    return ThreadORM(
        thread_id=_THREAD_ID,
        user_id=_USER.identity,
        status="idle",
        metadata_json={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _request(if_not_exists: str | None) -> RunCreate:
    return RunCreate(assistant_id="agent", input={"x": 1}, if_not_exists=if_not_exists, on_completion="keep")


async def _call(endpoint: str, request: RunCreate, *, thread: ThreadORM | None, stateless: bool = False) -> None:
    """Run one handler with the thread lookup answering ``thread``, stopping at ``_prepare_run``."""
    session = AsyncMock()
    session.scalar.return_value = thread
    session_scope = MagicMock()
    session_scope.__aenter__ = AsyncMock(return_value=session)
    session_scope.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aegra_api.api.runs._get_session_maker", return_value=MagicMock(return_value=session_scope)),
        patch("aegra_api.api.runs._apply_create_run_auth", new_callable=AsyncMock),
        patch("aegra_api.api.runs._prepare_run", new_callable=AsyncMock, side_effect=_ReachedPrepare),
    ):
        if stateless and endpoint == "create":
            await stateless_create_run(request, _USER, session)
        elif stateless and endpoint == "stream":
            await stateless_stream_run(request, _USER)
        elif stateless:
            await stateless_wait_for_run(request, _USER)
        elif endpoint == "create":
            await create_run(_THREAD_ID, request, _USER, session)
        elif endpoint == "stream":
            await create_and_stream_run(_THREAD_ID, request, _USER)
        else:
            await wait_for_run(_THREAD_ID, request, _USER)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
async def test_reject_returns_404_when_the_thread_is_missing(endpoint: str) -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(endpoint, _request("reject"), thread=None)

    assert exc.value.status_code == 404
    assert exc.value.detail == f"Thread '{_THREAD_ID}' not found"


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
async def test_reject_starts_the_run_on_an_existing_thread(endpoint: str) -> None:
    with pytest.raises(_ReachedPrepare):
        await _call(endpoint, _request("reject"), thread=_own_thread())


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("if_not_exists", [None, "create"])
async def test_a_missing_thread_is_created_unless_reject_is_asked(endpoint: str, if_not_exists: str | None) -> None:
    with pytest.raises(_ReachedPrepare):
        await _call(endpoint, _request(if_not_exists), thread=None)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
async def test_stateless_run_ignores_reject(endpoint: str) -> None:
    """A stateless run gets a fresh thread, so there is no missing thread to reject."""
    with pytest.raises(_ReachedPrepare):
        await _call(endpoint, _request("reject"), thread=None, stateless=True)
