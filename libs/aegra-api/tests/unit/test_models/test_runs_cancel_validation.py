"""RunsCancel accepts exactly one selector: status, or thread_id plus run_ids."""

import pytest
from pydantic import ValidationError

from aegra_api.models.runs import RunsCancel


class TestRunsCancelSelector:
    def test_status_selector_alone_is_valid(self) -> None:
        request = RunsCancel(status="pending")

        assert request.status == "pending"
        assert request.thread_id is None

    def test_thread_and_run_ids_selector_is_valid(self) -> None:
        request = RunsCancel(thread_id="t1", run_ids=["r1", "r2"])

        assert request.run_ids == ["r1", "r2"]

    def test_empty_body_is_invalid(self) -> None:
        with pytest.raises(ValidationError, match="either 'status' or both"):
            RunsCancel()

    def test_run_ids_without_thread_is_invalid(self) -> None:
        with pytest.raises(ValidationError, match="either 'status' or both"):
            RunsCancel(run_ids=["r1"])

    def test_both_selectors_is_invalid(self) -> None:
        with pytest.raises(ValidationError, match="either 'status' or both"):
            RunsCancel(status="all", thread_id="t1", run_ids=["r1"])

    @pytest.mark.parametrize("status", ["success", "interrupted", "bogus"])
    def test_non_active_status_is_invalid(self, status: str) -> None:
        with pytest.raises(ValidationError):
            RunsCancel(status=status)
