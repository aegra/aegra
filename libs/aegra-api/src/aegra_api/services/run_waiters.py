"""Heartbeat keep-alive utilities for join/wait endpoints.

Provides an async generator that streams periodic ``\\n`` heartbeat bytes
to keep HTTP connections alive through proxies and load balancers, then
yields the final JSON result when the run completes.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from sqlalchemy import select

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.services.executor import executor
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Terminal run states — used by join/wait to skip waiting
TERMINAL_STATES = {"success", "error", "interrupted"}

# The key the LangGraph SDK looks for in a wait/join body: `runs.wait()` raises
# on it by default. Same payload shape as the SSE `error` event that
# `streaming_service.signal_run_error` emits.
ERROR_KEY = "__error__"


def error_envelope(error: str, message: str) -> dict[str, Any]:
    """Build the body that reports a run did not produce a result."""
    return {ERROR_KEY: {"error": error, "message": message}}


def run_result_body(run_orm: RunORM | None, run_id: str, *, timed_out: bool = False) -> dict[str, Any]:
    """The join/wait response body for a run in its current state.

    ``success`` and ``interrupted`` return the run's output. An interrupt is a
    human-review pause rather than a failure, and its partial output is the
    whole point of the call. Every other state has no output worth returning —
    a failed run finalizes with ``{}`` — so the body says why instead.
    """
    if run_orm is None:
        return error_envelope("RunNotFound", f"Run '{run_id}' no longer exists")
    if run_orm.status == "error":
        return error_envelope("Error", run_orm.error_message or "Run failed")
    if run_orm.status == "timeout":
        return error_envelope("TimeoutError", run_orm.error_message or f"Run '{run_id}' timed out")
    if run_orm.status in TERMINAL_STATES:
        return run_orm.output or {}
    if timed_out:
        return error_envelope("TimeoutError", f"Run '{run_id}' did not finish within the wait timeout")
    return error_envelope(
        "IncompleteRun",
        f"Wait ended before run '{run_id}' completed (status: {run_orm.status})",
    )


async def read_run_result(
    run_id: str,
    thread_id: str,
    user_id: str,
    *,
    timed_out: bool = False,
) -> dict[str, Any]:
    """Open a short-lived DB session and build the run's final response body."""
    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == run_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user_id,
            )
        )
    return run_result_body(run_orm, run_id, timed_out=timed_out)


def encode_output(output: dict[str, Any]) -> bytes:
    """Serialize a run output dict to JSON bytes."""
    return json.dumps(output, default=str).encode()


async def heartbeat_wait_body(
    run_id: str,
    thread_id: str,
    user_id: str,
    *,
    timeout: float,
) -> AsyncIterator[bytes]:
    """Async generator that keeps the HTTP connection alive while waiting.

    Yields ``b"\\n"`` heartbeat bytes every ``KEEPALIVE_INTERVAL_SECS``
    until the run finishes, then yields the JSON result. Leading whitespace
    is ignored by JSON parsers so clients parse the concatenated body normally.
    """
    done = asyncio.Event()
    timed_out = False

    async def _wait_for_run() -> None:
        nonlocal timed_out
        try:
            await executor.wait_for_completion(run_id, timeout=timeout)
        except TimeoutError:
            timed_out = True
            logger.warning("heartbeat_wait timeout", run_id=run_id, timeout=timeout)
        except Exception:
            logger.exception("heartbeat_wait error", run_id=run_id)
        finally:
            done.set()

    task = asyncio.create_task(_wait_for_run())
    interval = settings.app.KEEPALIVE_INTERVAL_SECS
    try:
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=interval)
            except TimeoutError:
                yield b"\n"
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        else:
            # Surface any exception so asyncio doesn't warn about
            # "Task exception was never retrieved"
            if task.exception() is not None:
                logger.error(
                    "heartbeat_wait task failed",
                    run_id=run_id,
                    exc_info=task.exception(),
                )

    result = await read_run_result(run_id, thread_id, user_id, timed_out=timed_out)
    yield encode_output(result)
