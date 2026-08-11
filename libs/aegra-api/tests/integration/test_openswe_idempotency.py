"""OpenSWE's exact threaded-background idempotency contract."""

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aegra_api.core.orm import Assistant, Run, Thread
from aegra_api.models import Run as RunModel
from aegra_api.models import RunCreate, User
from aegra_api.models.run_job import RunJob
from aegra_api.services.run_preparation import _idempotent_run_id, _prepare_run
from aegra_api.services.worker_executor import WorkerExecutor
from aegra_api.settings import settings


@pytest.mark.asyncio
async def test_concurrent_same_key_submits_once_and_mismatch_conflicts() -> None:
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL is required for the OpenSWE PostgreSQL integration test")

    engine = create_async_engine(settings.db.database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    user = User(identity="openswe-idempotency", scopes=[])
    thread_id, assistant_id = "openswe-idempotency-thread", "openswe-idempotency-assistant"
    run_id = _idempotent_run_id(user.identity, thread_id, "delivery-1")
    try:
        async with engine.begin() as connection:
            await connection.execute(select(1))
        async with maker() as session:
            await session.execute(delete(Run).where(Run.thread_id == thread_id))
            await session.execute(delete(Thread).where(Thread.thread_id == thread_id))
            await session.execute(delete(Assistant).where(Assistant.assistant_id == assistant_id))
            session.add(
                Assistant(
                    assistant_id=assistant_id,
                    graph_id="contract-graph",
                    config={},
                    context={},
                    user_id=user.identity,
                    name="contract",
                    description="contract",
                    metadata_dict={},
                    version=1,
                )
            )
            session.add(Thread(thread_id=thread_id, user_id=user.identity, status="idle", metadata_json={}))
            await session.commit()

        async def create(
            payload: str | None,
            key: str | None = "delivery-1",
            command: dict | None = None,
        ) -> tuple[str, RunModel, RunJob]:
            async with maker() as session:
                return await _prepare_run(
                    session,
                    thread_id,
                    RunCreate(
                        assistant_id=assistant_id,
                        input={"message": payload} if command is None else None,
                        command=command,
                    ),
                    user,
                    initial_status="pending",
                    idempotency_key=key,
                )

        submit = AsyncMock()
        graph_service = MagicMock()
        graph_service.list_graphs.return_value = ["contract-graph"]
        with (
            patch("aegra_api.services.run_preparation.get_langgraph_service", return_value=graph_service),
            patch("aegra_api.services.run_preparation.executor.submit", submit),
        ):
            first, second = await asyncio.gather(create("same"), create("same"))
            assert first[0] == second[0] == run_id
            assert submit.await_count == 1
            async with maker() as session:
                count = await session.scalar(select(func.count()).select_from(Run).where(Run.run_id == run_id))
                assert count == 1
                await session.execute(update(Run).where(Run.run_id == run_id).values(status="success"))
                await session.execute(
                    update(Thread)
                    .where(Thread.thread_id == thread_id)
                    .values(status="idle", metadata_json={"sentinel": "unchanged"})
                )
                await session.commit()
            with pytest.raises(HTTPException) as conflict:
                await create("different")
            assert conflict.value.status_code == 409
            assert submit.await_count == 1
            async with maker() as session:
                thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
                assert thread is not None
                assert thread.status == "idle"
                assert thread.metadata_json == {"sentinel": "unchanged"}

            lost_insert = await create("lost-insert", "delivery-lost-insert")
            async with maker() as session:
                await session.execute(update(Run).where(Run.run_id == lost_insert[0]).values(status="success"))
                await session.execute(
                    update(Thread)
                    .where(Thread.thread_id == thread_id)
                    .values(status="idle", metadata_json={"sentinel": "lost-insert"})
                )
                await session.commit()
            submitted_after_lost_insert = submit.await_count
            async with maker() as session:
                original_scalar = session.scalar
                hide_existing_once = True

                async def hide_existing_run(statement: Any, *args: Any, **kwargs: Any) -> Any:
                    nonlocal hide_existing_once
                    if hide_existing_once:
                        hide_existing_once = False
                        return None
                    return await original_scalar(statement, *args, **kwargs)

                with patch.object(session, "scalar", AsyncMock(side_effect=hide_existing_run)):
                    lost_insert_replay = await _prepare_run(
                        session,
                        thread_id,
                        RunCreate(assistant_id=assistant_id, input={"message": "lost-insert"}),
                        user,
                        initial_status="pending",
                        idempotency_key="delivery-lost-insert",
                    )
            assert lost_insert_replay[0] == lost_insert[0]
            assert submit.await_count == submitted_after_lost_insert
            async with maker() as session:
                thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
                assert thread is not None
                assert thread.status == "idle"
                assert thread.metadata_json == {"sentinel": "lost-insert"}

            failed_id = _idempotent_run_id(user.identity, thread_id, "delivery-4")
            with (
                patch(
                    "aegra_api.services.run_preparation.executor.submit", AsyncMock(side_effect=RuntimeError("crash"))
                ),
                pytest.raises(RuntimeError, match="crash"),
            ):
                await create("recover", "delivery-4")
            with patch("aegra_api.services.worker_executor._get_session_maker", return_value=maker):
                executor = WorkerExecutor()
                assert await executor._poll_postgres(idempotent_only=True) == failed_id
                ordinary_id = (await create("ordinary", None))[0]
                async with maker() as session:
                    await session.execute(update(Run).where(Run.run_id == failed_id).values(status="error"))
                    await session.commit()
                assert await executor._poll_postgres(idempotent_only=True) is None
                assert await executor._poll_postgres() == ordinary_id

            submitted_before_resume = submit.await_count
            async with maker() as session:
                await session.execute(update(Thread).where(Thread.thread_id == thread_id).values(status="interrupted"))
                await session.commit()
            resumed = await create(None, "delivery-resume", command={"resume": "continue"})
            async with maker() as session:
                await session.execute(update(Run).where(Run.run_id == resumed[0]).values(status="success"))
                await session.execute(update(Thread).where(Thread.thread_id == thread_id).values(status="idle"))
                await session.commit()
            replayed = await create(None, "delivery-resume", command={"resume": "continue"})
            assert replayed[0] == resumed[0]
            assert submit.await_count == submitted_before_resume + 1
    finally:
        async with maker() as session:
            await session.execute(delete(Run).where(Run.thread_id == thread_id))
            await session.execute(delete(Thread).where(Thread.thread_id == thread_id))
            await session.execute(delete(Assistant).where(Assistant.assistant_id == assistant_id))
            await session.commit()
        await engine.dispose()
