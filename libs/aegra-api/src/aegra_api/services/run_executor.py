"""Graph run execution logic.

Single source of truth for executing a graph run. Both LocalExecutor
(asyncio.create_task) and WorkerExecutor (Redis BLPOP) call
`execute_run`. All database and broker interactions are delegated to
`run_status` and `streaming_service` respectively.
"""

import asyncio
from typing import Any

import structlog

from aegra_api.core.active_runs import active_runs
from aegra_api.core.auth_ctx import with_auth_ctx
from aegra_api.core.redis_manager import redis_manager
from aegra_api.models.run_job import RunJob
from aegra_api.observability.metrics import get_worker_metrics
from aegra_api.services.broker import broker_manager
from aegra_api.services.graph_streaming import stream_graph_events
from aegra_api.services.langgraph_service import create_run_config, get_langgraph_service
from aegra_api.services.run_status import finalize_run, update_run_status
from aegra_api.services.streaming_service import streaming_service
from aegra_api.settings import settings
from aegra_api.utils.run_utils import map_command_to_langgraph

logger = structlog.getLogger(__name__)

_DEFAULT_STREAM_MODES = ["values"]

# Run IDs whose cancellation was triggered by lease loss (not user action).
# When a heartbeat detects lease loss, it adds the run_id here before
# cancelling the job task. execute_run's CancelledError handler checks
# this set to skip finalize_run and SSE signaling — the reaper has already
# re-enqueued the run and another worker will execute it. Without this,
# the old worker would write status="interrupted" and send an SSE end event,
# prematurely closing client streams and potentially overwriting the new
# worker's status.
_lease_loss_cancellations: set[str] = set()

# Run IDs whose cancellation was triggered by a user action (API cancel
# endpoint). When the cancel endpoint fires, it adds the run_id here
# before calling task.cancel(). execute_run's CancelledError handler
# checks this set to finalize as "interrupted" and increment the
# completed metric. Without this, the default CancelledError path
# (timeout) would skip finalize — correct for timeouts where
# _execute_and_release handles finalization, but wrong for user cancels.
_user_cancellations: set[str] = set()


def mark_user_cancellation(run_id: str) -> None:
    """Mark a run as user-cancelled before task.cancel().

    Called by the cancel API endpoint so execute_run's CancelledError
    handler can distinguish user cancels from timeout cancels.
    """
    _user_cancellations.add(run_id)


def unmark_user_cancellation(run_id: str) -> None:
    """Remove a run from the user-cancellation set.

    Called by the cancel endpoint if the cancel/interrupt signal fails,
    to prevent orphaning the entry in the set.
    """
    _user_cancellations.discard(run_id)


def _increment_completed(graph_id: str, status: str) -> None:
    """Increment runs_completed counter. No-op if metrics not enabled."""
    metrics = get_worker_metrics()
    if metrics is not None:
        metrics.runs_completed.labels(graph_id=graph_id, status=status).inc()


async def execute_run(job: RunJob) -> None:
    """Execute a graph run, stream events to the broker, and update DB.

    Handles the full lifecycle: status transitions, event streaming,
    interrupt detection, cancellation, and error signaling.

    CancelledError discrimination (three sources):
    - **Lease-loss**: heartbeat adds run_id to ``_lease_loss_cancellations``
      before cancelling — skip finalize, reaper re-enqueues.
    - **User cancel**: API adds run_id to ``_user_cancellations`` before
      cancelling — finalize as interrupted, increment completed metric.
    - **Timeout** (default): ``asyncio.wait_for`` cancels — skip finalize,
      ``_execute_and_release`` handles finalization as error.
    """
    run_id = job.identity.run_id
    thread_id = job.identity.thread_id
    graph_id = job.identity.graph_id
    is_lease_loss = False

    try:
        await update_run_status(run_id, "running")

        final_output = await _stream_graph(job)

        if final_output.has_interrupt:
            await finalize_run(
                run_id,
                thread_id,
                status="interrupted",
                thread_status="interrupted",
                output=final_output.data,
            )
            _increment_completed(graph_id, "interrupted")
        else:
            await finalize_run(
                run_id,
                thread_id,
                status="success",
                thread_status="idle",
                output=final_output.data,
            )
            _increment_completed(graph_id, "success")

    except asyncio.CancelledError:
        if run_id in _lease_loss_cancellations:
            # Lease was lost — the reaper re-enqueued this run for another
            # worker.  Do NOT finalize, signal done, or clean up the broker.
            # The new worker owns the run now.
            is_lease_loss = True
            logger.info("Lease-loss cancel, skipping finalize", run_id=run_id)
        elif run_id in _user_cancellations:
            # User cancel — finalize as interrupted
            await finalize_run(run_id, thread_id, status="interrupted", thread_status="idle", output={})
            _increment_completed(graph_id, "interrupted")
            await _best_effort_signal(streaming_service.signal_run_cancelled, run_id)
        else:
            # Timeout (default) — skip finalize. _execute_and_release
            # handles it: finalize(error) + completed{error} + run_timeouts.
            pass
        raise
    except Exception as exc:
        logger.exception("Run failed", run_id=run_id)
        safe_message = f"{type(exc).__name__}: execution failed"
        await finalize_run(run_id, thread_id, status="error", thread_status="error", output={}, error=str(exc))
        _increment_completed(graph_id, "error")
        await _best_effort_signal(streaming_service.signal_run_error, run_id, safe_message, type(exc).__name__)
    else:
        status = "interrupted" if final_output.has_interrupt else "success"
        await _best_effort_signal(_signal_end_event, run_id, status)
    finally:
        _lease_loss_cancellations.discard(run_id)
        _user_cancellations.discard(run_id)
        active_runs.pop(run_id, None)
        if not is_lease_loss:
            await streaming_service.cleanup_run(run_id)
            await _signal_run_done(run_id)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


async def _best_effort_signal(fn: Any, *args: Any) -> None:
    """Call a signaling function, logging but not raising on failure.

    Signaling (end events, cancel/error notifications) must not override
    an already-committed DB status if it fails.
    """
    try:
        await fn(*args)
    except Exception:
        logger.warning("Signal failed (best-effort, DB status already committed)", fn=fn.__name__)


class _GraphResult:
    """Accumulates output and interrupt state during graph streaming."""

    __slots__ = ("data", "has_interrupt")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.has_interrupt: bool = False


async def _stream_graph(job: RunJob) -> _GraphResult:
    """Load the graph, stream events to the broker, return final output."""
    run_id = job.identity.run_id
    run_config = _build_run_config(job)
    execution_input = _resolve_input(job)
    stream_modes = _resolve_stream_modes(job.execution.stream_mode)

    langgraph_service = get_langgraph_service()
    result = _GraphResult()

    async with (
        langgraph_service.get_graph(
            job.identity.graph_id,
            config=run_config,
            access_context="threads.create_run",
            user=job.user,
            context=job.execution.context,
        ) as graph,
        with_auth_ctx(job.user, job.user.permissions),  # type: ignore[arg-type]
    ):
        async for event_type, event_data in stream_graph_events(
            graph=graph,
            input_data=execution_input,
            config=run_config,
            stream_mode=stream_modes,
            context=job.execution.context,
            subgraphs=job.behavior.subgraphs,
            on_checkpoint=lambda _: None,
            on_task_result=lambda _: None,
        ):
            event_id = await broker_manager.allocate_event_id(run_id)
            await streaming_service.put_to_broker(run_id, event_id, (event_type, event_data))

            if isinstance(event_data, dict) and "__interrupt__" in event_data:
                result.has_interrupt = True
            if event_type.startswith("values"):
                result.data = event_data

    return result


def _build_run_config(job: RunJob) -> dict[str, Any]:
    """Assemble the LangGraph run config from a RunJob."""
    config = create_run_config(
        job.identity.run_id,
        job.identity.thread_id,
        job.user,
        additional_config=job.execution.config,
        checkpoint=job.execution.checkpoint,
    )
    if job.behavior.interrupt_before is not None:
        items = job.behavior.interrupt_before
        config["interrupt_before"] = items if isinstance(items, list) else [items]
    if job.behavior.interrupt_after is not None:
        items = job.behavior.interrupt_after
        config["interrupt_after"] = items if isinstance(items, list) else [items]
    return config


def _resolve_input(job: RunJob) -> Any:
    """Return graph input — either raw data or a LangGraph Command."""
    if job.execution.command is not None:
        return map_command_to_langgraph(job.execution.command)
    return job.execution.input_data


def _resolve_stream_modes(stream_mode: str | list[str] | None) -> list[str]:
    """Normalize stream_mode to a list."""
    if stream_mode is None:
        return _DEFAULT_STREAM_MODES.copy()
    if isinstance(stream_mode, str):
        return [stream_mode]
    return list(stream_mode)


# ------------------------------------------------------------------
# End event (tells SSE consumers the stream is finished)
# ------------------------------------------------------------------


async def _signal_end_event(run_id: str, status: str) -> None:
    """Publish an 'end' event to the broker so SSE consumers close cleanly.

    Without this, the SSE connection hangs after the last data event
    because the broker never signals completion on success/interrupt.
    Error and cancel paths already send end events via signal_run_error
    and signal_run_cancelled respectively.
    """
    broker = broker_manager.get_broker(run_id)
    if broker is None or broker.is_finished():
        return

    event_id = await broker_manager.allocate_event_id(run_id)
    await broker.put(event_id, ("end", {"status": status}))


# ------------------------------------------------------------------
# Done signal (Redis key for fast completion polling)
# ------------------------------------------------------------------

_DONE_KEY_TTL_SECONDS = 3600


async def _signal_run_done(run_id: str) -> None:
    """Set a Redis key indicating the run has finished.

    WorkerExecutor.wait_for_completion polls this key instead of
    subscribing to the broker's event channel. Simpler, no subscription
    race, no message parsing. Falls back silently if Redis unavailable.
    """
    try:
        client = redis_manager.get_client()
        done_key = f"{settings.redis.REDIS_CHANNEL_PREFIX}done:{run_id}"
        await client.set(done_key, "1", ex=_DONE_KEY_TTL_SECONDS)
    except Exception:
        logger.debug("Redis done-key set failed (non-critical)", run_id=run_id)
