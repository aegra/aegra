"""SQL-level deep-copy of a thread row + its checkpoint history.

One Postgres transaction on the checkpointer pool at REPEATABLE READ:
no cross-pool coordination, no orphan rows on failure, snapshot pinned
against concurrent writers. Column lists resolved via
``pg_catalog.pg_attribute`` + ``to_regclass`` so introspection follows
the same ``search_path`` as the unqualified DML (forward-compat with
upstream column additions).
"""

from typing import TYPE_CHECKING, Any

import structlog
from psycopg import sql as pgsql
from psycopg.types.json import Jsonb

from aegra_api.core.database import db_manager

if TYPE_CHECKING:
    from psycopg import AsyncConnection

logger = structlog.getLogger(__name__)

# Tables managed by langgraph-checkpoint-postgres. Each has ``thread_id``
# as a non-PK leading column; the copy preserves all other columns verbatim.
_CHECKPOINT_TABLES: tuple[str, ...] = ("checkpoints", "checkpoint_writes", "checkpoint_blobs")


async def _table_columns(conn: "AsyncConnection", table: str) -> list[str]:
    """Return live column names of ``table`` via ``pg_attribute``+``to_regclass``.

    Filters dropped-column tombstones (``attisdropped``); ``AS column_name``
    matches the pool's ``dict_row`` factory used by callers.
    """
    cur = await conn.execute(
        "SELECT attname AS column_name FROM pg_catalog.pg_attribute "
        "WHERE attrelid = to_regclass(%s) "
        "AND attnum > 0 AND NOT attisdropped "
        "ORDER BY attnum",
        (table,),
    )
    rows = await cur.fetchall()
    return [r["column_name"] for r in rows]


async def _copy_checkpoint_table(
    conn: "AsyncConnection",
    table: str,
    src_thread_id: str,
    new_thread_id: str,
) -> None:
    """Copy ``src_thread_id`` rows of ``table`` under ``new_thread_id``.

    Raises on missing ``thread_id`` so the outer transaction rolls back;
    identifiers composed via ``psycopg.sql.Identifier`` (input from
    ``pg_attribute`` may contain quotes that break f-string quoting).
    """
    cols = await _table_columns(conn, table)
    if "thread_id" not in cols:
        raise RuntimeError(f"Table {table!r} has no 'thread_id' column; aborting copy to preserve atomicity")
    other_cols = [c for c in cols if c != "thread_id"]
    table_id = pgsql.Identifier(table)
    tid_id = pgsql.Identifier("thread_id")
    cols_composed = pgsql.SQL(", ").join(pgsql.Identifier(c) for c in other_cols)
    stmt = pgsql.SQL("INSERT INTO {table} ({tid}, {cols}) SELECT %s, {cols} FROM {table} WHERE thread_id = %s").format(
        table=table_id, tid=tid_id, cols=cols_composed
    )
    await conn.execute(stmt, (new_thread_id, src_thread_id))


async def copy_thread_atomically(
    *,
    src_thread_id: str,
    new_thread_id: str,
    src_status: str,
    src_metadata: dict[str, Any],
    user_identity: str,
) -> None:
    """Insert thread row + copy checkpoint history in one REPEATABLE READ tx.

    ``metadata["owner"]`` is rewritten to ``user_identity`` (mirrors
    ``create_thread``), timestamps regenerated to copy time, status
    inherited from source.
    """
    if db_manager.lg_pool is None:
        raise RuntimeError("Checkpoint pool is not initialized")

    # Shallow copy: only top-level ``owner`` mutates; ``or {}`` defends None.
    metadata = dict(src_metadata or {})
    metadata["owner"] = user_identity

    async with db_manager.lg_pool.connection() as conn, conn.transaction():
        # First statement in tx — required for Postgres to honour the level.
        await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        await conn.execute(
            'INSERT INTO "thread" ("thread_id", "status", "metadata_json", "user_id", '
            '"created_at", "updated_at") '
            "VALUES (%s, %s, %s, %s, NOW(), NOW())",
            (new_thread_id, src_status, Jsonb(metadata), user_identity),
        )
        for table in _CHECKPOINT_TABLES:
            await _copy_checkpoint_table(conn, table, src_thread_id, new_thread_id)
    # Audit pair: API layer logs on failure via ``logger.exception``.
    logger.info(
        "thread.copy",
        src_thread_id=src_thread_id,
        new_thread_id=new_thread_id,
        user_identity=user_identity,
    )
