"""Tests for the process-local run cancellation registry."""

from unittest.mock import MagicMock, patch

from aegra_api.core.active_runs import request_local_cancellation


def test_request_local_cancellation_cancels_live_task() -> None:
    mock_task = MagicMock()
    mock_task.done.return_value = False
    run_id = "run-123"

    with (
        patch.dict("aegra_api.core.active_runs.active_runs", {run_id: mock_task}, clear=True),
        patch("aegra_api.core.active_runs.explicit_run_cancellations", set()) as cancellations,
    ):
        assert request_local_cancellation(run_id) is True

    mock_task.cancel.assert_called_once()
    assert run_id in cancellations


def test_request_local_cancellation_marks_intent_without_local_task() -> None:
    run_id = "run-123"

    with (
        patch.dict("aegra_api.core.active_runs.active_runs", {}, clear=True),
        patch("aegra_api.core.active_runs.explicit_run_cancellations", set()) as cancellations,
    ):
        assert request_local_cancellation(run_id) is False

    assert run_id in cancellations


def test_request_local_cancellation_skips_completed_task() -> None:
    mock_task = MagicMock()
    mock_task.done.return_value = True
    run_id = "run-123"

    with (
        patch.dict("aegra_api.core.active_runs.active_runs", {run_id: mock_task}, clear=True),
        patch("aegra_api.core.active_runs.explicit_run_cancellations", set()) as cancellations,
    ):
        assert request_local_cancellation(run_id) is False

    mock_task.cancel.assert_not_called()
    assert run_id in cancellations
