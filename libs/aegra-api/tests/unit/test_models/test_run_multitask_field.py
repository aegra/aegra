"""The run record reports the strategy it was admitted with, as the SDK's Run type declares."""

from typing import Any

from aegra_api.models.runs import Run
from tests.fixtures.test_helpers import make_run_orm


def _orm_run(execution_params: dict[str, Any] | None) -> Any:
    return make_run_orm(execution_params=execution_params)


def test_strategy_is_read_back_from_execution_params() -> None:
    run = Run.model_validate(_orm_run({"behavior": {"multitask_strategy": "rollback"}}))
    assert run.multitask_strategy == "rollback"


def test_default_strategy_when_none_was_requested() -> None:
    # The gate applied enqueue, and so did every run created before the field existed.
    assert Run.model_validate(_orm_run({"behavior": {"multitask_strategy": None}})).multitask_strategy == "enqueue"
    assert Run.model_validate(_orm_run(None)).multitask_strategy == "enqueue"


def test_legacy_unvalidated_value_reports_the_default() -> None:
    # Before the field was a Literal, any string was stored; such a row must not 500 the run endpoints.
    assert Run.model_validate(_orm_run({"behavior": {"multitask_strategy": "cancel"}})).multitask_strategy == "enqueue"
