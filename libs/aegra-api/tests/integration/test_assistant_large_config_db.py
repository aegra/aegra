"""Regression tests for assistant configs larger than PostgreSQL btree entries."""

import hashlib
from uuid import uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.settings import settings

_ASSISTANT_ID = "large-config-index-regression"
_USER_ID = "large-config-test-user"
_GRAPH_ID = "large-config-test-graph"


def _large_system_prompt() -> str:
    """Return deterministic, incompressible-ish text above the btree tuple limit."""
    return "\n".join(hashlib.sha256(f"prompt-line-{i}".encode()).hexdigest() for i in range(512))


@pytest.mark.asyncio
async def test_insert_assistant_with_large_configurable_prompt_succeeds() -> None:
    """A large prompt in config.configurable must not overflow the unique index."""
    engine = create_async_engine(settings.db.database_url)

    try:
        assistant_table = None
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                assistant_table = await conn.scalar(text("SELECT to_regclass('public.assistant')"))
        except (OperationalError, OSError) as exc:
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")

        if assistant_table is None:
            pytest.skip("assistant table is unavailable; run Alembic migrations before this DB regression test")

        prompt = _large_system_prompt()
        config = {
            "configurable": {
                "system_prompt": prompt,
                "request_id": str(uuid4()),
            }
        }

        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with Session() as session:
            await session.execute(delete(AssistantORM).where(AssistantORM.assistant_id == _ASSISTANT_ID))
            await session.commit()

            assistant = AssistantORM(
                assistant_id=_ASSISTANT_ID,
                name="Large Config Regression",
                description="Exercises the md5(config::text) uniqueness index",
                graph_id=_GRAPH_ID,
                config=config,
                context={},
                user_id=_USER_ID,
                metadata_dict={"test": "large-config-index"},
                version=1,
            )
            session.add(assistant)
            await session.commit()

            try:
                saved = await session.get(AssistantORM, _ASSISTANT_ID)
                assert saved is not None
                assert saved.config["configurable"]["system_prompt"] == prompt
            finally:
                await session.execute(delete(AssistantORM).where(AssistantORM.assistant_id == _ASSISTANT_ID))
                await session.commit()
    finally:
        await engine.dispose()
