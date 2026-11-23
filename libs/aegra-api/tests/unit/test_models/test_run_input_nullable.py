"""Regression: Run.input must accept None for checkpoint-only resume."""

from datetime import UTC, datetime
from typing import Any

import pytest

from aegra_api.models.runs import Run


def test_run_accepts_none_input() -> None:
    """Run rows from checkpoint-only resume have input=None in the ORM.

    Regression: Run.input was non-nullable, causing 500s when serializing
    resumed runs back through GET /runs/{id} or POST /runs/wait.
    """
    now = datetime.now(UTC)
    run = Run(
        run_id="r-1",
        thread_id="t-1",
        assistant_id="agent",
        status="pending",
        input=None,
        user_id="u-1",
        created_at=now,
        updated_at=now,
    )
    assert run.input is None


@pytest.mark.parametrize("value", [["a", "b"], "hello", 7, True, {"x": 1}])
def test_run_returns_any_json_input(value: Any) -> None:
    """A run created with a list or scalar input must serialize back, not 500 on the response model."""
    now = datetime.now(UTC)

    run = Run(
        run_id="r-1",
        thread_id="t-1",
        assistant_id="agent",
        status="pending",
        input=value,
        user_id="u-1",
        created_at=now,
        updated_at=now,
    )

    assert run.model_dump()["input"] == value


@pytest.mark.parametrize("value", [["a", "b"], "done", 7, False, {"count": 2}])
def test_run_returns_any_json_output(value: Any) -> None:
    """A graph may return a list or a scalar; the run row stores it, so the response must carry it."""
    now = datetime.now(UTC)

    run = Run(
        run_id="r-1",
        thread_id="t-1",
        assistant_id="agent",
        status="success",
        input=None,
        output=value,
        user_id="u-1",
        created_at=now,
        updated_at=now,
    )

    assert run.model_dump()["output"] == value
