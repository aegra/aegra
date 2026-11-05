"""Semantic store tests that need a real PostgreSQL with pgvector.

They build the store from a ``store.index`` whose embed is a file path, as the server
does, in a schema of their own so the database's store tables keep their dimensions.
They skip when no database is reachable.
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from aegra_api.core.store_index import resolve_store_index
from aegra_api.settings import settings

EMBED_FILE = """
VOCABULARY = ("short", "answers", "dark", "mode", "python", "code")


async def embed(texts: list[str]) -> list[list[float]]:
    return [[float(text.lower().count(word)) for word in VOCABULARY] for text in texts]
"""


@pytest.fixture
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    # psycopg's async mode cannot run on the Proactor loop Windows uses by default.
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def schema_pool() -> AsyncIterator[AsyncConnectionPool[psycopg.AsyncConnection[DictRow]]]:
    """A pool whose connections default to a throwaway schema, or skip without a database."""
    try:
        admin = await psycopg.AsyncConnection.connect(settings.db.database_url_sync, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")

    schema = f"store_index_test_{uuid4().hex[:12]}"
    async with admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        pool = AsyncConnectionPool(
            conninfo=settings.db.database_url_sync,
            connection_class=psycopg.AsyncConnection[DictRow],
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,
                "row_factory": dict_row,
                "options": f"-c search_path={schema},public",
            },
        )
        await pool.open()
        try:
            yield pool
        finally:
            await pool.close()
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


async def test_store_searches_with_embed_function_loaded_from_path(
    schema_pool: AsyncConnectionPool[psycopg.AsyncConnection[DictRow]], tmp_path: Path
) -> None:
    (tmp_path / "embeddings.py").write_text(EMBED_FILE)
    index = resolve_store_index(
        {"dims": 6, "embed": "./embeddings.py:embed", "fields": ["text"]},
        base_dir=tmp_path,
    )
    store = AsyncPostgresStore(conn=schema_pool, index=index)
    await store.setup()

    await store.aput(("docs",), "style", {"text": "The user wants short answers"})
    await store.aput(("docs",), "theme", {"text": "The user prefers dark mode"})
    results = await store.asearch(("docs",), query="short answers, please", limit=2)

    assert [item.key for item in results] == ["style", "theme"]
    assert results[0].score is not None
    assert results[1].score is not None
    assert results[0].score > results[1].score
