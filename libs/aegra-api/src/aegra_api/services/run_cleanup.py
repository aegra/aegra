"""Helpers for deleting ephemeral threads after stateless runs complete."""

import asyncio

import structlog
from sqlalchemy import select

from aegra_api.core.active_runs import active_runs
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.services.executor import executor
from aegra_api.services.streaming_service import streaming_service

logger = structlog.getLogger(__name__)

_background_cleanup_tasks: set[asyncio.Task[None]] = set()


async def delete_thread_by_id(thread_id: str, user_id: str) -> None:
    """Delete an ephemeral thread and cascade-delete any runs on it."""
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
            await streaming_service.cancel_run(run_id)
            task = active_runs.pop(run_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Error awaiting cancelled task during thread cleanup", run_id=run_id)

        thread = await session.scalar(
            select(ThreadORM).where(
                ThreadORM.thread_id == thread_id,
                ThreadORM.user_id == user_id,
            )
        )
        if thread:
            await session.delete(thread)
            await session.commit()


async def cleanup_after_background_run(run_id: str, thread_id: str, user_id: str) -> None:
    """Wait for a background run to finish, then delete its ephemeral thread."""
    try:
        await executor.wait_for_completion(run_id, timeout=3600.0)
    except (asyncio.CancelledError, TimeoutError):
        pass
    except Exception:
        logger.exception("Error waiting for background run", run_id=run_id)

    try:
        await delete_thread_by_id(thread_id, user_id)
    except Exception:
        logger.exception("Failed to delete ephemeral thread", thread_id=thread_id, run_id=run_id)


def schedule_background_cleanup(run_id: str, thread_id: str, user_id: str) -> asyncio.Task[None]:
    """Keep a strong reference to cleanup tasks until they finish."""
    task = asyncio.create_task(cleanup_after_background_run(run_id, thread_id, user_id))
    _background_cleanup_tasks.add(task)
    task.add_done_callback(_background_cleanup_tasks.discard)
    return task
