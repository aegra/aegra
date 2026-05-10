"""Unit tests for ``services/run_cancellation``.

Covers the helpers extracted from ``cancel_run_endpoint``:
``try_cancel_pending`` (atomic CAS + metric increment) and
``signal_user_cancel`` (registry tag + broadcast with rollback on failure).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import prometheus_client
import pytest
from redis import RedisError

from aegra_api.core.cancellation_state import cancellations
from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import get_worker_metrics, setup_worker_metrics
from aegra_api.services.run_cancellation import (
    signal_user_cancel,
    try_cancel_pending,
)


@pytest.fixture(autouse=True)
def _reset_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level metric singletons before each test.

    ``fresh_registry`` only gives each test a new ``CollectorRegistry``;
    without this reset the cached ``_worker_metrics`` / ``_reaper_metrics``
    objects (bound to the previous test's registry) keep their counter
    values and a CAS-hit increment in one test bleeds into the next.
    """
    monkeypatch.setattr(metrics_module, "_worker_metrics", None)
    monkeypatch.setattr(metrics_module, "_reaper_metrics", None)


@pytest.fixture
def fresh_registry() -> prometheus_client.CollectorRegistry:
    """Isolated Prometheus registry so each test starts with zeroed counters."""
    return prometheus_client.CollectorRegistry()


def _make_run(run_id: str, graph_id: str | None) -> MagicMock:
    """Build a Run-shaped MagicMock with the fields try_cancel_pending reads."""
    run_orm = MagicMock()
    run_orm.run_id = run_id
    run_orm.execution_params = {"graph_id": graph_id} if graph_id is not None else None
    return run_orm


def _make_session(rowcount: int) -> MagicMock:
    """Mock AsyncSession whose execute() returns a Result with the given rowcount."""
    session = MagicMock()

    async def execute(_stmt: object) -> Any:
        result = MagicMock()
        result.rowcount = rowcount
        return result

    session.execute = AsyncMock(side_effect=execute)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_try_cancel_pending_returns_true_and_increments_metric_on_cas_hit(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """CAS hit → commit, log, increment runs_completed{status='interrupted'},
    emit terminal cancel signal so streaming clients see the end event,
    return True."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    run_orm = _make_run("run-cas-hit", graph_id="graph-cas")
    session = _make_session(rowcount=1)

    with patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming:
        mock_streaming.signal_run_cancelled = AsyncMock()
        result = await try_cancel_pending(session, run_orm)

    assert result is True
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()
    mock_streaming.signal_run_cancelled.assert_awaited_once_with("run-cas-hit")
    assert metrics.runs_completed.labels(graph_id="graph-cas", status="interrupted")._value.get() == 1.0


@pytest.mark.asyncio
async def test_try_cancel_pending_returns_false_and_rolls_back_on_cas_miss(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """CAS miss (worker won the race) → rollback, NO metric increment,
    NO terminal signal (the worker that picked the run up owns those),
    return False."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    run_orm = _make_run("run-cas-miss", graph_id="graph-cas")
    session = _make_session(rowcount=0)

    with patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming:
        mock_streaming.signal_run_cancelled = AsyncMock()
        result = await try_cancel_pending(session, run_orm)

    assert result is False
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()
    mock_streaming.signal_run_cancelled.assert_not_awaited()
    # The success-path metric must stay at zero — the worker that picked
    # the run up will increment it via execute_run.
    assert metrics.runs_completed.labels(graph_id="graph-cas", status="interrupted")._value.get() == 0.0


@pytest.mark.asyncio
async def test_try_cancel_pending_handles_missing_graph_id_in_params(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """``execution_params=None`` falls back to graph_id='unknown' on the metric."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    run_orm = _make_run("run-no-params", graph_id=None)
    session = _make_session(rowcount=1)

    with patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming:
        mock_streaming.signal_run_cancelled = AsyncMock()
        result = await try_cancel_pending(session, run_orm)

    assert result is True
    assert metrics.runs_completed.labels(graph_id="unknown", status="interrupted")._value.get() == 1.0


@pytest.mark.asyncio
async def test_try_cancel_pending_swallows_redis_error_from_signal(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """If the broker is unreachable, the cancel must still succeed — the DB
    row is already 'interrupted' and clients will eventually time out
    rather than wait for an end event that will never arrive."""
    setup_worker_metrics(registry=fresh_registry)

    run_orm = _make_run("run-broker-down", graph_id="graph-cas")
    session = _make_session(rowcount=1)

    with patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming:
        mock_streaming.signal_run_cancelled = AsyncMock(side_effect=RedisError("redis down"))
        result = await try_cancel_pending(session, run_orm)

    assert result is True
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_signal_user_cancel_marks_then_clears_when_broadcast_raises_redis_error() -> None:
    """If the streaming broadcast raises RedisError, the 'user' tag must be
    rolled back so a later lease-loss tag isn't shadowed and the registry
    doesn't grow unbounded under repeated broker outages."""
    run_id = "run-broadcast-fail"
    cancellations.clear(run_id)

    with patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming:
        mock_streaming.cancel_run = AsyncMock(side_effect=RedisError("redis down"))
        mock_streaming.interrupt_run = AsyncMock()

        with pytest.raises(RedisError):
            await signal_user_cancel(run_id, "cancel")

    # 'user' tag must have been cleared after the failed broadcast.
    assert cancellations.reason_of(run_id) is None


@pytest.mark.asyncio
async def test_signal_user_cancel_preserves_lease_loss_tag_on_broadcast_failure() -> None:
    """If a lease-loss tag was set between mark and broadcast failure,
    rollback uses ``only='user'`` so the lease-loss tag survives."""
    run_id = "run-mixed-tags"
    cancellations.clear(run_id)

    async def fake_cancel(_run_id: str) -> None:
        # Simulate a lease-loss tag arriving from the heartbeat between
        # the user-mark and the broadcast outcome.
        cancellations.mark(run_id, "lease_loss")
        raise RedisError("redis down mid-broadcast")

    try:
        with patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming:
            mock_streaming.cancel_run = AsyncMock(side_effect=fake_cancel)
            mock_streaming.interrupt_run = AsyncMock()

            with pytest.raises(RedisError):
                await signal_user_cancel(run_id, "cancel")
            # After rollback (only='user'), the lease_loss tag set during
            # broadcast must survive — otherwise a concurrent lease loss
            # would be silently demoted to a default-timeout classification.
            assert cancellations.reason_of(run_id) == "lease_loss"
    finally:
        cancellations.clear(run_id)


@pytest.mark.asyncio
async def test_signal_user_cancel_routes_interrupt_to_streaming_interrupt() -> None:
    """``action='interrupt'`` calls streaming_service.interrupt_run, not cancel_run."""
    run_id = "run-interrupt"
    cancellations.clear(run_id)

    try:
        with patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming:
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.interrupt_run = AsyncMock()

            await signal_user_cancel(run_id, "interrupt")

            mock_streaming.interrupt_run.assert_awaited_once_with(run_id)
            mock_streaming.cancel_run.assert_not_awaited()
    finally:
        cancellations.clear(run_id)
