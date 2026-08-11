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
from aegra_api.services.run_status import finalize_run
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
    foreign_run_id = "openswe-idempotency-foreign-run"
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
            await session.flush()
            session.add(
                Run(
                    run_id=foreign_run_id,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    status="running",
                    input={},
                    config={},
                    context={},
                    output=None,
                    error_message=None,
                    user_id="openswe-idempotency-other-tenant",
                    execution_params={},
                )
            )
            await session.commit()

        async def create(
            payload: str | None,
            key: str | None = "delivery-1",
            strategy: str | None = None,
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
                        multitask_strategy=strategy,
                    ),
                    user,
                    initial_status="pending",
                    idempotency_key=key,
                    openswe_background_admission=True,
                )

        events: list[tuple[str, str]] = []
        submit = AsyncMock(side_effect=lambda job: events.append(("submit", job.identity.run_id)))

        async def acknowledge_interrupt(old: str) -> bool:
            events.append(("interrupt", old))
            async with maker() as ack_session:
                await ack_session.execute(update(Run).where(Run.run_id == old).values(status="interrupted"))
                await ack_session.commit()
            events.append(("ack", old))
            return True

        interrupt = AsyncMock(side_effect=acknowledge_interrupt)
        graph_service = MagicMock()
        graph_service.list_graphs.return_value = ["contract-graph"]
        with (
            patch("aegra_api.services.run_preparation.get_langgraph_service", return_value=graph_service),
            patch("aegra_api.services.run_preparation.executor.submit", submit),
            patch("aegra_api.services.run_preparation.streaming_service.interrupt_run", interrupt),
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
                        openswe_background_admission=True,
                    )
            assert lost_insert_replay[0] == lost_insert[0]
            assert submit.await_count == submitted_after_lost_insert
            async with maker() as session:
                thread = await session.scalar(select(Thread).where(Thread.thread_id == thread_id))
                assert thread is not None
                assert thread.status == "idle"
                assert thread.metadata_json == {"sentinel": "lost-insert"}
                await session.execute(update(Run).where(Run.run_id == run_id).values(status="running"))
                await session.commit()

            replacement_keys = ("delivery-2a", "delivery-2b")
            replacements = await asyncio.gather(*(create("replacement", key, "interrupt") for key in replacement_keys))
            assert interrupt.await_count == 1
            async with maker() as session:
                states = [await session.scalar(select(Run.status).where(Run.run_id == run[0])) for run in replacements]
            assert sorted(states) == ["interrupted", "pending"]
            async with maker() as session:
                foreign_status = await session.scalar(select(Run.status).where(Run.run_id == foreign_run_id))
            assert foreign_status == "running"
            pending_index = states.index("pending")
            replacement = replacements[pending_index]
            submitted_after_replacements = submit.await_count
            await create("replacement", replacement_keys[pending_index], "interrupt")
            assert interrupt.await_count == 1
            assert submit.await_count == submitted_after_replacements

            async with maker() as session:
                await session.execute(update(Run).where(Run.run_id == replacement[0]).values(status="running"))
                await session.commit()
            with (
                patch(
                    "aegra_api.services.run_preparation.streaming_service.interrupt_run", AsyncMock(return_value=True)
                ),
                patch("aegra_api.services.run_preparation._INTERRUPT_SETTLE_ATTEMPTS", 1),
                patch("aegra_api.services.run_preparation._INTERRUPT_SETTLE_INTERVAL_SECONDS", 0),
                pytest.raises(HTTPException, match="acknowledge") as timeout,
            ):
                await create("blocked", "delivery-3", "interrupt")
            assert timeout.value.status_code == 409
            assert submit.await_count == submitted_after_replacements

            race_id = _idempotent_run_id(user.identity, thread_id, "terminal-race")
            async with maker() as session:
                await session.execute(update(Run).where(Run.run_id == replacement[0]).values(status="pending"))
                await session.commit()
                original_scalar = session.scalar
                complete_before_claim = True

                async def complete_pending_run(statement: Any, *args: Any, **kwargs: Any) -> Any:
                    nonlocal complete_before_claim
                    if complete_before_claim and getattr(statement, "is_update", False):
                        complete_before_claim = False
                        async with maker() as competing_session:
                            await competing_session.execute(
                                update(Run).where(Run.run_id == replacement[0]).values(status="success")
                            )
                            await competing_session.commit()
                    return await original_scalar(statement, *args, **kwargs)

                with (
                    patch.object(session, "scalar", AsyncMock(side_effect=complete_pending_run)),
                    pytest.raises(HTTPException) as terminal_race,
                ):
                    await _prepare_run(
                        session,
                        thread_id,
                        RunCreate(
                            assistant_id=assistant_id,
                            input={"message": "terminal-race"},
                            multitask_strategy="interrupt",
                        ),
                        user,
                        initial_status="pending",
                        idempotency_key="terminal-race",
                        openswe_background_admission=True,
                    )
                assert terminal_race.value.status_code == 409
                assert not complete_before_claim
                assert await session.scalar(select(Run.run_id).where(Run.run_id == race_id)) is None
                await session.execute(update(Run).where(Run.run_id == replacement[0]).values(status="running"))
                await session.commit()
            assert submit.await_count == submitted_after_replacements

            with patch("aegra_api.services.run_status._get_session_maker", return_value=maker):
                await finalize_run(
                    replacement[0], thread_id, user_id=user.identity, status="success", thread_status="idle"
                )
                await finalize_run(run_id, thread_id, user_id=user.identity, status="error", thread_status="error")
            async with maker() as session:
                assert await session.scalar(select(Run.status).where(Run.run_id == run_id)) == "interrupted"
                assert await session.scalar(select(Run.status).where(Run.run_id == replacement[0])) == "success"
                assert await session.scalar(select(Thread.status).where(Thread.thread_id == thread_id)) == "idle"

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
