"""Heartbeat keep-alive utilities for join/wait endpoints.

Provides an async generator that streams periodic ``\\n`` heartbeat bytes
to keep HTTP connections alive through proxies and load balancers, then
yields the final JSON result when the run completes.
"""

import asyncio
import contextlib
import json
import time
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
CANCEL_WAIT_POLL_SECONDS = 0.5
CANCEL_WAIT_SLACK_SECONDS = 5.0


def cancel_wait_timeout_seconds() -> float:
    """Bound for cancel?wait=1: two heartbeats plus slack."""
    return 2 * settings.worker.HEARTBEAT_INTERVAL_SECONDS + CANCEL_WAIT_SLACK_SECONDS


async def wait_for_terminal_run(
    run_id: str,
    *,
    user_id: str,
    timeout_seconds: float | None = None,
) -> RunORM | None:
    """Poll the run row until it is terminal or the wait bound elapses.

    Opens a short-lived session per poll so the cancel request does not hold
    a connection for the full wait.
    """
    maker = _get_session_maker()
    deadline = time.monotonic() + (timeout_seconds if timeout_seconds is not None else cancel_wait_timeout_seconds())
    latest: RunORM | None = None
    while True:
        async with maker() as session:
            latest = await session.scalar(
                select(RunORM).where(
                    RunORM.run_id == run_id,
                    RunORM.user_id == user_id,
                )
            )
            if latest is not None:
                session.expunge(latest)
                if latest.status in TERMINAL_STATES:
                    return latest
        if time.monotonic() >= deadline:
            return latest
        await asyncio.sleep(CANCEL_WAIT_POLL_SECONDS)


async def read_run_output(
    run_id: str,
    thread_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Open a short-lived DB session and read the run's final output."""
    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == run_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user_id,
            )
        )
        if not run_orm:
            return {}
        return run_orm.output or {}


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

    async def _wait_for_run() -> None:
        try:
            await executor.wait_for_completion(run_id, timeout=timeout)
        except TimeoutError:
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

    output = await read_run_output(run_id, thread_id, user_id)
    yield encode_output(output)
