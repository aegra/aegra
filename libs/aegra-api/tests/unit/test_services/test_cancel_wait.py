"""Tests for cancel?wait=1 terminal polling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.run_waiters import cancel_wait_timeout_seconds, wait_for_terminal_run


def test_cancel_wait_timeout_derives_from_heartbeat() -> None:
    with patch("aegra_api.services.run_waiters.settings") as mock_settings:
        mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 10
        assert cancel_wait_timeout_seconds() == 25.0


@pytest.mark.asyncio
async def test_wait_for_terminal_run_returns_on_interrupted() -> None:
    run = MagicMock()
    run.status = "interrupted"
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=run)
    session.expunge = MagicMock()
    maker = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker.return_value = ctx

    with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
        result = await wait_for_terminal_run("run-1", user_id="user-1", timeout_seconds=5)

    assert result is run
    session.expunge.assert_called_once_with(run)


@pytest.mark.asyncio
async def test_wait_for_terminal_run_times_out_still_running() -> None:
    run = MagicMock()
    run.status = "running"
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=run)
    session.expunge = MagicMock()
    maker = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker.return_value = ctx

    with (
        patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker),
        patch("aegra_api.services.run_waiters.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("aegra_api.services.run_waiters.time") as mock_time,
    ):
        mock_time.monotonic.side_effect = [0.0, 6.0]
        result = await wait_for_terminal_run("run-1", user_id="user-1", timeout_seconds=5)

    assert result is run
    mock_sleep.assert_not_awaited()
