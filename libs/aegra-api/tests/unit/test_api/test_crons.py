"""Unit tests for cron API helpers."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aegra_api.api.crons import _trigger_first_run
from aegra_api.models import Run, User


def _make_cron(
    *,
    assistant_id: str = "agent",
    thread_id: str | None = None,
    on_run_completed: str | None = None,
    payload: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cron_id="cron-001",
        assistant_id=assistant_id,
        thread_id=thread_id,
        on_run_completed=on_run_completed,
        payload=payload
        if payload is not None
        else {"input": {"messages": [{"role": "user", "content": "hi"}]}, "config": {"k": "v"}},
    )


class TestTriggerFirstRun:
    """Test initial run triggering for cron creation."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def mock_run(self, mock_user: User) -> Run:
        return Run(
            run_id="run-001",
            thread_id="thread-001",
            assistant_id="agent",
            status="pending",
            input={"messages": [{"role": "user", "content": "hi"}]},
            user_id=mock_user.identity,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @pytest.mark.asyncio
    async def test_schedules_cleanup_for_stateless_cron(
        self, mock_user: User, mock_session: AsyncMock, mock_run: Run
    ) -> None:
        cron = _make_cron(thread_id=None, on_run_completed=None)

        with (
            patch("aegra_api.api.crons.uuid4", return_value="eph-thread-1"),
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                return_value=("run-001", mock_run, Mock()),
            ) as mock_prepare,
            patch("aegra_api.api.crons.schedule_background_cleanup") as mock_schedule,
        ):
            result = await _trigger_first_run(mock_session, cron, mock_user)

        assert result is mock_run
        mock_prepare.assert_awaited_once()
        assert mock_prepare.await_args.args[1] == "eph-thread-1"
        mock_schedule.assert_called_once_with("run-001", "eph-thread-1", mock_user.identity)

    @pytest.mark.asyncio
    async def test_skips_cleanup_for_thread_bound_cron(
        self, mock_user: User, mock_session: AsyncMock, mock_run: Run
    ) -> None:
        cron = _make_cron(thread_id="thread-bound-1")

        with (
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                return_value=("run-001", mock_run, Mock()),
            ),
            patch("aegra_api.api.crons.schedule_background_cleanup") as mock_schedule,
        ):
            await _trigger_first_run(mock_session, cron, mock_user)

        mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_cleanup_when_keep_requested(
        self, mock_user: User, mock_session: AsyncMock, mock_run: Run
    ) -> None:
        cron = _make_cron(thread_id=None, on_run_completed="keep")

        with (
            patch("aegra_api.api.crons.uuid4", return_value="eph-thread-keep"),
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                return_value=("run-001", mock_run, Mock()),
            ),
            patch("aegra_api.api.crons.schedule_background_cleanup") as mock_schedule,
        ):
            await _trigger_first_run(mock_session, cron, mock_user)

        mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_deletes_ephemeral_thread_when_initial_run_setup_fails(
        self, mock_user: User, mock_session: AsyncMock
    ) -> None:
        cron = _make_cron(thread_id=None, on_run_completed=None)

        with (
            patch("aegra_api.api.crons.uuid4", return_value="eph-thread-fail"),
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("aegra_api.api.crons.delete_thread_by_id", new_callable=AsyncMock) as mock_delete,
            pytest.raises(RuntimeError, match="boom"),
        ):
            await _trigger_first_run(mock_session, cron, mock_user)

        mock_delete.assert_awaited_once_with("eph-thread-fail", mock_user.identity)
