"""Shared helpers for deleting ephemeral threads after stateless runs complete.

Consumed by stateless_runs.py and crons.py — both create short-lived threads
that need cleanup after the underlying run finishes.
"""

import asyncio

import structlog
from psycopg import Error as PsycopgError
from redis import RedisError
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.core.active_runs import active_runs
from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.services.executor import executor
from aegra_api.services.run_status import interrupt_run_before_delete
from aegra_api.services.streaming_service import streaming_service
from aegra_api.services.thread_hooks import notify_threads_deleted, thread_deleted_event

logger = structlog.getLogger(__name__)

# Strong refs so fire-and-forget cleanup tasks survive GC until done.
_background_cleanup_tasks: set[asyncio.Task[None]] = set()
# The run each cleanup task is still waiting for.
_waiting_for_run: dict[asyncio.Task[None], str] = {}
# Shutdown's answer per waiting cleanup task, asked once: whether its run had ended.
_run_ended_at_shutdown: dict[asyncio.Task[None], bool] = {}

# Transient infra failures we tolerate during cleanup. Programmer errors
# (TypeError, AttributeError, ...) propagate.
_CLEANUP_ERRORS: tuple[type[BaseException], ...] = (RedisError, SQLAlchemyError, OSError, PsycopgError)


async def delete_thread_by_id(thread_id: str, user_id: str) -> None:
    """Delete an ephemeral thread and cascade-delete its runs.

    Opens its own DB session so it can be called after the request session has
    been closed (e.g. in a finally block or background task).
    """
    maker = _get_session_maker()
    async with maker() as session:
        active_runs_stmt = select(RunORM).where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status.in_(["pending", "running"]),
        )
        active_runs_list = (await session.scalars(active_runs_stmt)).all()

        for run in active_runs_list:
            run_id = run.run_id
            await interrupt_run_before_delete(run)
            await streaming_service.cancel_run(run_id)
            task = active_runs.pop(run_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Expected: we just called task.cancel(). Nothing to log.
                    pass
                except _CLEANUP_ERRORS:
                    logger.exception("Error awaiting cancelled task during thread cleanup", run_id=run_id)

        thread = await session.scalar(
            select(ThreadORM).where(
                ThreadORM.thread_id == thread_id,
                ThreadORM.user_id == user_id,
            )
        )
        if thread:
            # Checkpoints first — same ordering rationale as the delete_thread route.
            await db_manager.get_checkpointer().adelete_thread(thread_id)
            event = thread_deleted_event(thread.thread_id, thread.user_id, thread.metadata_json)
            removed = await session.scalar(
                delete(ThreadORM)
                .where(ThreadORM.thread_id == thread.thread_id, ThreadORM.user_id == thread.user_id)
                .returning(ThreadORM.thread_id)
            )
            await session.commit()
            # No row back means a concurrent delete committed first and has already reported the thread.
            if removed is not None:
                notify_threads_deleted([event])


async def cleanup_after_background_run(run_id: str, thread_id: str, user_id: str) -> None:
    """Wait for a background run to finish, then delete its ephemeral thread.

    executor.wait_for_completion works both in-process (dev) and cross-instance
    (prod with Redis workers).
    """
    task = asyncio.current_task()
    if task is not None:
        _waiting_for_run[task] = run_id
    try:
        await executor.wait_for_completion(run_id, timeout=3600.0)
    except (asyncio.CancelledError, TimeoutError):
        # Cancellation = shutdown; timeout = run exceeded 1h cap. Either way we
        # still proceed to delete the thread below — no need to log.
        pass
    except _CLEANUP_ERRORS:
        logger.exception("Error waiting for background run", run_id=run_id)
    finally:
        if task is not None:
            _waiting_for_run.pop(task, None)

    try:
        await delete_thread_by_id(thread_id, user_id)
    except _CLEANUP_ERRORS:
        logger.exception("Failed to delete ephemeral thread", thread_id=thread_id, run_id=run_id)


async def cleanup_tasks_for_shutdown() -> list[asyncio.Task[None]]:
    """Cleanup tasks shutdown waits for, so the threads they delete still reach the thread-deleted hook.

    A task past its run wait is always included. A task still waiting is included only when the executor
    reports its run finished, since its wait then returns and it goes on to delete; a task whose run is
    still going is left running, and the thread it deletes later gets no hook call. Each waiting task's run
    is checked once per shutdown, and a check that fails counts the run as finished.
    """
    tasks: list[asyncio.Task[None]] = []
    for task in list(_background_cleanup_tasks):
        run_id = _waiting_for_run.get(task)
        if run_id is None or await _run_ended(task, run_id):
            tasks.append(task)
    return tasks


async def _run_ended(task: asyncio.Task[None], run_id: str) -> bool:
    ended = _run_ended_at_shutdown.get(task)
    if ended is None:
        try:
            ended = await executor.is_finished(run_id)
        except _CLEANUP_ERRORS:
            logger.exception("Could not check a stateless run at shutdown; waiting for its cleanup", run_id=run_id)
            ended = True
        _run_ended_at_shutdown[task] = ended
    return ended


def _forget(task: asyncio.Task[None]) -> None:
    _background_cleanup_tasks.discard(task)
    _run_ended_at_shutdown.pop(task, None)


def schedule_background_cleanup(run_id: str, thread_id: str, user_id: str) -> asyncio.Task[None]:
    """Fire-and-forget background cleanup, strong ref held until done."""
    task = asyncio.create_task(cleanup_after_background_run(run_id, thread_id, user_id))
    _background_cleanup_tasks.add(task)
    task.add_done_callback(_forget)
    return task
