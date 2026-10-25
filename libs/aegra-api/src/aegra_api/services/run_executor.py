"""Graph run execution logic.

Single source of truth for executing a graph run. Both LocalExecutor
(asyncio.create_task) and WorkerExecutor (Redis BLPOP) call
`execute_run`. All database and broker interactions are delegated to
`run_status` and `streaming_service` respectively.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog

from aegra_api.core.active_runs import active_runs, executing_runs, reconciled_run_cancellations
from aegra_api.core.auth_ctx import with_auth_ctx
from aegra_api.core.redis_manager import redis_manager
from aegra_api.models.run_job import RunJob
from aegra_api.services.broker import broker_manager
from aegra_api.services.event_streaming.native_stream import stream_native_v3_events
from aegra_api.services.graph_streaming import stream_graph_events
from aegra_api.services.langgraph_service import create_run_config, create_thread_config, get_langgraph_service
from aegra_api.services.run_status import finalize_run, start_run
from aegra_api.services.streaming_service import streaming_service
from aegra_api.services.webhook_delivery import RunOutcome, notify_run_finished
from aegra_api.settings import settings
from aegra_api.utils.run_utils import map_command_to_langgraph

logger = structlog.getLogger(__name__)

_DEFAULT_STREAM_MODES = ["values"]

# Cancellation provenance prevents infrastructure stops from looking user-initiated.
_lease_loss_cancellations: set[str] = set()
_shutdown_cancellations: set[str] = set()
_timeout_cancellations: set[str] = set()

_TIMEOUT_ERROR = "Job exceeded maximum execution time"
_TIMEOUT_SAFE_MESSAGE = "TimeoutError: execution failed"


async def execute_run(job: RunJob) -> None:
    """Execute a graph run, stream events to the broker, and update DB.

    Handles the full lifecycle: status transitions, event streaming,
    interrupt detection, cancellation, and error signaling.
    """
    run_id = job.identity.run_id
    thread_id = job.identity.thread_id
    user_id = job.user.identity
    resumes_elsewhere = False
    finalized = False
    # What the completion webhook reports; partial values survive a failed run.
    result = _GraphResult()
    final_status = "error"
    error_info: dict[str, str] | None = None
    started_at: datetime | None = None
    executing_runs.add(run_id)

    try:
        if not await start_run(run_id, user_id=user_id):
            logger.info("Run became terminal before execution started", run_id=run_id)
            return
        started_at = datetime.now(UTC)

        final_output = await _stream_graph(job, result)

        if final_output.has_interrupt:
            final_status = "interrupted"
            finalized = await finalize_run(
                run_id,
                thread_id,
                user_id=user_id,
                status="interrupted",
                thread_status="interrupted",
                output=final_output.data,
            )
        else:
            final_status = "success"
            finalized = await finalize_run(
                run_id,
                thread_id,
                user_id=user_id,
                status="success",
                thread_status="idle",
                output=final_output.data,
            )

    except asyncio.CancelledError:
        if run_id in _lease_loss_cancellations:
            resumes_elsewhere = True
            logger.info("Lease-loss cancel, skipping finalize", run_id=run_id)
        elif run_id in _shutdown_cancellations:
            # Drain cancel: the run goes back to the queue, so finalizing here
            # would destroy work a plain crash (SIGKILL) recovers (#474).
            resumes_elsewhere = True
            logger.info("Shutdown drain cancel, skipping finalize for requeue", run_id=run_id)
        elif run_id in _timeout_cancellations:
            final_status = "error"
            error_info = {"error": "TimeoutError", "message": _TIMEOUT_ERROR}
            finalized = await finalize_run(
                run_id,
                thread_id,
                user_id=user_id,
                status="error",
                thread_status="error",
                output={},
                error=_TIMEOUT_ERROR,
            )
            if finalized:
                await _best_effort_signal(
                    streaming_service.signal_run_error,
                    run_id,
                    _TIMEOUT_SAFE_MESSAGE,
                    "TimeoutError",
                )
        else:
            final_status = "interrupted"
            finalized = await finalize_run(
                run_id,
                thread_id,
                user_id=user_id,
                status="interrupted",
                thread_status="idle",
                output={},
            )
            if finalized:
                await _best_effort_signal(streaming_service.signal_run_cancelled, run_id)
        raise
    except Exception as exc:
        logger.exception("Run failed", run_id=run_id)
        safe_message = f"{type(exc).__name__}: execution failed"
        # Also reached when finalizing a success fails, so reset the status set above.
        final_status = "error"
        error_info = {"error": type(exc).__name__, "message": str(exc)}
        finalized = await finalize_run(
            run_id,
            thread_id,
            user_id=user_id,
            status="error",
            thread_status="error",
            output={},
            error=str(exc),
        )
        if finalized:
            await _best_effort_signal(streaming_service.signal_run_error, run_id, safe_message, type(exc).__name__)
    else:
        if finalized:
            status = "interrupted" if final_output.has_interrupt else "success"
            await _best_effort_signal(_signal_end_event, run_id, status)
    finally:
        _lease_loss_cancellations.discard(run_id)
        _shutdown_cancellations.discard(run_id)
        _timeout_cancellations.discard(run_id)
        active_runs.pop(run_id, None)
        # No await between this check and leaving executing_runs, so the API either handed the run over or reports it.
        handed_off = run_id in reconciled_run_cancellations
        reconciled_run_cancellations.discard(run_id)
        executing_runs.discard(run_id)
        if finalized:
            notify_run_finished(
                job,
                RunOutcome(status=final_status, values=result.state, error=error_info, run_started_at=started_at),
            )
        elif handed_off:
            notify_run_finished(job, RunOutcome(status="interrupted", values=result.state, run_started_at=started_at))
        if not resumes_elsewhere:
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

    __slots__ = ("data", "has_interrupt", "state")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.has_interrupt: bool = False
        # Graph state for the completion webhook; None until one is seen or read.
        self.state: dict[str, Any] | None = None


def _is_state(data: Any) -> bool:
    # Interrupt notices are re-emitted as values events but carry no state.
    return isinstance(data, dict) and set(data) != {"__interrupt__"}


async def _read_final_state(graph: Any, job: RunJob) -> dict[str, Any] | None:
    """The thread's latest checkpointed values, whichever stream modes the client chose."""
    try:
        snapshot = await graph.aget_state(create_thread_config(job.identity.thread_id, job.user))
    except Exception:
        # Best effort: the webhook falls back to the values streamed so far.
        logger.warning("Could not read the final state for the run webhook", run_id=job.identity.run_id, exc_info=True)
        return None
    return snapshot.values if snapshot is not None else None


async def _stream_graph(job: RunJob, result: _GraphResult | None = None) -> _GraphResult:
    """Load the graph, stream events to the broker, return final output.

    Pass *result* to keep the values streamed so far when the run fails midway.
    """
    run_config = _build_run_config(job)
    execution_input = _resolve_input(job)
    stream_modes = _resolve_stream_modes(job.execution.stream_mode)

    langgraph_service = get_langgraph_service()
    result = result if result is not None else _GraphResult()

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
        if job.execution.event_streaming_v2:
            await _stream_native_v2(job, graph, execution_input, run_config, result)
        else:
            await _stream_legacy(job, graph, execution_input, run_config, stream_modes, result)
        if job.behavior.webhook:
            final_state = await _read_final_state(graph, job)
            if final_state is not None:
                result.state = final_state

    return result


async def _stream_legacy(
    job: RunJob,
    graph: Any,
    execution_input: Any,
    run_config: dict[str, Any],
    stream_modes: list[str],
    result: _GraphResult,
) -> None:
    """Stream via the v1 producer (legacy SSE endpoints)."""
    run_id = job.identity.run_id
    async for event_type, event_data in stream_graph_events(
        graph=graph,
        input_data=execution_input,
        config=run_config,
        stream_mode=stream_modes,
        context=job.execution.context,
        subgraphs=job.behavior.subgraphs,
        durability=job.execution.durability,
        on_checkpoint=lambda _: None,
        on_task_result=lambda _: None,
    ):
        event_id = await broker_manager.allocate_event_id(run_id)
        await streaming_service.put_to_broker(run_id, event_id, (event_type, event_data))

        if isinstance(event_data, dict) and "__interrupt__" in event_data:
            result.has_interrupt = True
        if event_type.startswith("values"):
            result.data = event_data
        if event_type == "values" and _is_state(event_data):
            result.state = event_data


async def _stream_native_v2(
    job: RunJob,
    graph: Any,
    execution_input: Any,
    run_config: dict[str, Any],
    result: _GraphResult,
) -> None:
    """Stream via the native v3 protocol producer (Agent Protocol v2).

    Each native event goes into the broker as ``(method, protocol_event)``.
    Interrupts ride on a ``values`` event's ``params.interrupts``; the final
    state is the last ``values`` event's ``params.data``.
    """
    run_id = job.identity.run_id
    async for method, event in stream_native_v3_events(
        graph=graph,
        input_data=execution_input,
        config=run_config,
        context=job.execution.context,
        durability=job.execution.durability,
    ):
        event_id = await broker_manager.allocate_event_id(run_id)
        await streaming_service.put_to_broker(run_id, event_id, (method, event))

        params = event.get("params", {})
        data = params.get("data")
        # Interrupts ride on values.params.interrupts, or as an __interrupt__ key
        # in a values/updates payload (the path session.py routes to input.requested).
        if params.get("interrupts") or (isinstance(data, dict) and "__interrupt__" in data):
            result.has_interrupt = True
        if method == "values" and isinstance(data, dict):
            result.data = data
            if _is_state(data):
                result.state = data


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
