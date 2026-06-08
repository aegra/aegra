"""Integration tests for run cancellation.

Tests verify that cancelling a run properly cancels the asyncio task
and sets the run status to 'interrupted'.

Addresses GitHub Issue #132: Cancel endpoint doesn't cancel asyncio task
"""

import asyncio
import contextlib
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from aegra_api.core.active_runs import active_runs
from aegra_api.core.cancellation_state import cancellations
from aegra_api.services.broker import RunBroker, broker_manager
from aegra_api.services.redis_broker import RedisBrokerManager
from aegra_api.services.streaming_service import streaming_service


@pytest.mark.asyncio
class TestCancelRun:
    """Test run cancellation properly cancels asyncio tasks"""

    @pytest.fixture
    def run_id(self) -> str:
        return str(uuid4())

    async def test_cancel_already_completed_run_returns_true(self, run_id: str) -> None:
        """Test that cancelling an already completed run doesn't error"""
        # No task registered for this run_id
        result = await streaming_service.cancel_run(run_id)
        # Should return True even if no task to cancel
        assert result is True

    async def test_cancel_run_cancels_running_task(self, run_id: str) -> None:
        """Test that cancel_run actually cancels the asyncio task"""
        task_was_cancelled = False

        async def cancellable_task() -> None:
            nonlocal task_was_cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                task_was_cancelled = True
                raise

        task = asyncio.create_task(cancellable_task())
        active_runs[run_id] = task

        # Create a broker for the run
        broker = RunBroker(run_id)
        if not hasattr(broker_manager, "_brokers"):
            broker_manager._brokers = {}
        broker_manager._brokers[run_id] = broker

        # Cancel the run
        result = await streaming_service.cancel_run(run_id)
        assert result is True

        # Wait for task to be cancelled
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert task_was_cancelled or task.cancelled(), "Task should have been cancelled"

        # Cleanup
        active_runs.pop(run_id, None)
        broker_manager._brokers.pop(run_id, None)

    async def test_interrupt_run_cancels_running_task(self, run_id: str) -> None:
        """Test that interrupt_run actually cancels the asyncio task"""
        task_was_cancelled = False

        async def cancellable_task() -> None:
            nonlocal task_was_cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                task_was_cancelled = True
                raise

        task = asyncio.create_task(cancellable_task())
        active_runs[run_id] = task

        broker = RunBroker(run_id)
        if not hasattr(broker_manager, "_brokers"):
            broker_manager._brokers = {}
        broker_manager._brokers[run_id] = broker

        # Interrupt the run
        result = await streaming_service.interrupt_run(run_id)
        assert result is True

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert task_was_cancelled or task.cancelled(), "Task should have been cancelled"

        active_runs.pop(run_id, None)
        broker_manager._brokers.pop(run_id, None)


@pytest.mark.asyncio
class TestCancelRunStatusNotOverwritten:
    """Test that cancelled run status is not overwritten to 'success'

    When a task is cancelled via asyncio.Task.cancel(), it will raise
    CancelledError which is caught by execute_run_async and the status
    is set to 'interrupted'. The normal completion path (which would
    set status to 'success') is never reached.
    """

    @pytest.fixture
    def run_id(self) -> str:
        return str(uuid4())

    async def test_cancelled_task_raises_cancelled_error(self, run_id: str) -> None:
        """Verify that cancelled task properly raises CancelledError"""
        completed_normally = False
        was_cancelled = False

        async def task_with_completion_check() -> None:
            nonlocal completed_normally, was_cancelled
            try:
                await asyncio.sleep(10)
                completed_normally = True
            except asyncio.CancelledError:
                was_cancelled = True
                raise

        task = asyncio.create_task(task_with_completion_check())
        active_runs[run_id] = task

        broker = RunBroker(run_id)
        if not hasattr(broker_manager, "_brokers"):
            broker_manager._brokers = {}
        broker_manager._brokers[run_id] = broker

        # Let it start
        await asyncio.sleep(0.1)

        # Cancel using broker_manager
        await broker_manager.request_cancel(run_id, "cancel")

        # Wait for cancellation to propagate
        with pytest.raises(asyncio.CancelledError):
            await task

        assert was_cancelled, "Task should have received CancelledError"
        assert not completed_normally, "Task should NOT have completed normally"

        active_runs.pop(run_id, None)
        broker_manager._brokers.pop(run_id, None)


@pytest.mark.asyncio
class TestRedisCancelListenerMarksUserCancellation:
    """Multi-instance: when a cancel arrives via Redis pub/sub, the listener
    must mark the run as user-cancelled before task.cancel() so execute_run's
    CancelledError handler classifies it correctly (not as a timeout)."""

    async def test_execute_cancel_marks_before_cancelling(self) -> None:
        """Order check: cancellations.mark must run before task.cancel().

        Without this ordering, a cancel arriving on instance B via Redis
        pub/sub would have the run_id unmarked on B when execute_run's
        CancelledError handler runs, falling to the default (timeout)
        branch and skipping interrupted-finalize.
        """
        run_id = str(uuid4())
        call_order: list[str] = []

        # Replace the task with a mock so cancel() is observable
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock(side_effect=lambda: call_order.append("cancel"))
        active_runs[run_id] = mock_task

        try:
            mgr = RedisBrokerManager.__new__(RedisBrokerManager)
            # Use in-memory broker for the end event
            mgr.get_or_create_broker = broker_manager.get_or_create_broker  # type: ignore[method-assign]
            mgr.allocate_event_id = broker_manager.allocate_event_id  # type: ignore[method-assign]

            with patch.object(
                cancellations,
                "mark",
                side_effect=lambda rid, reason: call_order.append("mark"),
            ):
                await mgr._execute_cancel(run_id)

            assert call_order[:2] == ["mark", "cancel"], (
                f"cancellations.mark must precede task.cancel(); got order: {call_order}"
            )
        finally:
            active_runs.pop(run_id, None)
            cancellations.clear(run_id)
            broker_manager._brokers.pop(run_id, None)
