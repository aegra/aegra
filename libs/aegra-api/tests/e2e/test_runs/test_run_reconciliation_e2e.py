"""End-to-end coverage for terminal run and thread reconciliation."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.settings import settings
from tests.e2e._utils import elog

InterruptionEndpoint = Literal["cancel", "patch"]


@asynccontextmanager
async def _seed_run(
    *,
    run_status: str,
    thread_status: str,
    claimed_by: str | None = None,
    lease_expires_at: datetime | None = None,
    execution_params: dict[str, Any] | None = None,
    additional_run_status: str | None = None,
) -> AsyncIterator[tuple[str, str]]:
    engine = create_async_engine(settings.db.database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    assistant_id = str(uuid4())
    thread_id = str(uuid4())
    run_id = str(uuid4())
    now = datetime.now(UTC)

    async with maker() as session:
        session.add(
            AssistantORM(
                assistant_id=assistant_id,
                name="Run reconciliation test",
                graph_id="stress_test",
                config={"test_id": assistant_id},
                context={},
                user_id="anonymous",
                metadata_dict={},
                version=1,
                created_at=now,
            )
        )
        await session.flush()
        session.add(
            ThreadORM(
                thread_id=thread_id,
                status=thread_status,
                user_id="anonymous",
                metadata_json={},
                created_at=now,
                updated_at=now,
            )
        )
        await session.flush()
        session.add(
            RunORM(
                run_id=run_id,
                thread_id=thread_id,
                assistant_id=assistant_id,
                status=run_status,
                input={},
                user_id="anonymous",
                execution_params=execution_params,
                claimed_by=claimed_by,
                lease_expires_at=lease_expires_at,
                created_at=now,
                updated_at=now,
            )
        )
        if additional_run_status is not None:
            session.add(
                RunORM(
                    run_id=str(uuid4()),
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    status=additional_run_status,
                    input={},
                    user_id="anonymous",
                    execution_params=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        await session.commit()

    try:
        yield thread_id, run_id
    finally:
        async with maker() as session:
            await session.execute(delete(RunORM).where(RunORM.thread_id == thread_id))
            await session.execute(delete(ThreadORM).where(ThreadORM.thread_id == thread_id))
            await session.execute(delete(AssistantORM).where(AssistantORM.assistant_id == assistant_id))
            await session.commit()
        await engine.dispose()


async def _request_interruption(
    client: httpx.AsyncClient,
    endpoint: InterruptionEndpoint,
    thread_id: str,
    run_id: str,
) -> httpx.Response:
    if endpoint == "cancel":
        return await client.post(f"/threads/{thread_id}/runs/{run_id}/cancel")
    return await client.patch(
        f"/threads/{thread_id}/runs/{run_id}",
        json={"run_id": run_id, "status": "interrupted"},
    )


async def _read_state(thread_id: str, run_id: str) -> tuple[RunORM, ThreadORM]:
    engine = create_async_engine(settings.db.database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            run = await session.get(RunORM, run_id)
            thread = await session.get(ThreadORM, thread_id)
            assert run is not None
            assert thread is not None
            return run, thread
    finally:
        await engine.dispose()


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.parametrize("endpoint", ["cancel", "patch"])
async def test_stale_owner_interruption_reconciles_run_thread_and_lease(
    endpoint: InterruptionEndpoint,
) -> None:
    expired = datetime.now(UTC) - timedelta(minutes=1)
    async with _seed_run(
        run_status="running",
        thread_status="busy",
        claimed_by="dead-worker",
        lease_expires_at=expired,
    ) as (thread_id, run_id):
        async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=10.0) as client:
            response = await _request_interruption(client, endpoint, thread_id, run_id)

        response.raise_for_status()
        run, thread = await _read_state(thread_id, run_id)
        elog(
            "Stale-owner interruption state",
            {"endpoint": endpoint, "run_status": run.status, "thread_status": thread.status},
        )

        assert response.json()["status"] == "interrupted"
        assert run.status == "interrupted"
        assert run.claimed_by is None
        assert run.lease_expires_at is None
        assert thread.status == "idle"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.parametrize("endpoint", ["cancel", "patch"])
async def test_interruption_does_not_overwrite_terminal_run(endpoint: InterruptionEndpoint) -> None:
    async with _seed_run(run_status="success", thread_status="idle") as (thread_id, run_id):
        async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=10.0) as client:
            response = await _request_interruption(client, endpoint, thread_id, run_id)

        response.raise_for_status()
        run, thread = await _read_state(thread_id, run_id)
        elog(
            "Terminal interruption guard state",
            {"endpoint": endpoint, "run_status": run.status, "thread_status": thread.status},
        )

        assert response.json()["status"] == "success"
        assert run.status == "success"
        assert thread.status == "idle"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.parametrize("endpoint", ["cancel", "patch"])
async def test_interruption_keeps_thread_busy_when_another_run_is_active(
    endpoint: InterruptionEndpoint,
) -> None:
    async with _seed_run(
        run_status="pending",
        thread_status="busy",
        additional_run_status="running",
    ) as (thread_id, run_id):
        async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=10.0) as client:
            response = await _request_interruption(client, endpoint, thread_id, run_id)

        response.raise_for_status()
        run, thread = await _read_state(thread_id, run_id)
        elog(
            "Concurrent-run interruption state",
            {"endpoint": endpoint, "run_status": run.status, "thread_status": thread.status},
        )

        assert response.json()["status"] == "interrupted"
        assert run.status == "interrupted"
        assert thread.status == "busy"


@pytest.mark.e2e
@pytest.mark.prod_only
async def test_retry_exhaustion_marks_run_and_thread_error() -> None:
    expired = datetime.now(UTC) - timedelta(minutes=1)
    async with _seed_run(
        run_status="running",
        thread_status="busy",
        claimed_by="dead-worker",
        lease_expires_at=expired,
        execution_params={"_retry_count": settings.worker.BG_JOB_MAX_RETRIES},
    ) as (thread_id, run_id):
        deadline = asyncio.get_running_loop().time() + settings.worker.REAPER_INTERVAL_SECONDS * 2 + 5
        while asyncio.get_running_loop().time() < deadline:
            run, thread = await _read_state(thread_id, run_id)
            if run.status == "error":
                break
            await asyncio.sleep(0.5)

        elog("Retry exhaustion state", {"run_status": run.status, "thread_status": thread.status})
        assert run.status == "error"
        assert run.claimed_by is None
        assert run.lease_expires_at is None
        assert thread.status == "error"
