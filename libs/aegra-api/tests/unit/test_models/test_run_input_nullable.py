"""Regression: Run.input must accept None for checkpoint-only resume."""

from datetime import UTC, datetime

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
