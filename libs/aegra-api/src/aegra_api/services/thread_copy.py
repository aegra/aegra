"""SQL-level deep-copy of a thread row + its checkpoint history.

One Postgres transaction on the checkpointer pool at REPEATABLE READ:
no cross-pool coordination, no orphan rows on failure, snapshot pinned
against concurrent writers. Column lists resolved via
``pg_catalog.pg_attribute`` + ``to_regclass`` so the copy tracks live
schema changes instead of a hardcoded column list, following the same
``search_path`` as the unqualified DML.
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


async def _copy_thread_row(
    conn: "AsyncConnection",
    *,
    src_thread_id: str,
    new_thread_id: str,
    user_identity: str,
    metadata_overrides: dict[str, Any],
) -> None:
    """Insert the copy's ``thread`` row via ``INSERT ... SELECT`` from source.

    Columns are introspected so future additions are inherited verbatim.
    Overridden: identity + timestamps regenerated; active ``status``
    (busy/running) reset to idle since the copy has no ``runs`` row; and
    ``metadata_json`` merged source ``||`` handler ``||`` ``owner`` (caller).
    """
    cols = await _table_columns(conn, "thread")
    if "thread_id" not in cols:
        raise RuntimeError("Table 'thread' has no 'thread_id' column; aborting copy to preserve atomicity")

    merged_metadata = {**metadata_overrides, "owner": user_identity}
    select_parts: list[pgsql.Composable] = []
    params: list[Any] = []
    for col in cols:
        col_id = pgsql.Identifier(col)
        if col == "thread_id":
            select_parts.append(pgsql.Placeholder())
            params.append(new_thread_id)
        elif col == "user_id":
            select_parts.append(pgsql.Placeholder())
            params.append(user_identity)
        elif col == "status":
            select_parts.append(
                pgsql.SQL("CASE WHEN {c} IN ('busy', 'running') THEN 'idle' ELSE {c} END").format(c=col_id)
            )
        elif col == "metadata_json":
            select_parts.append(
                pgsql.SQL("COALESCE({c}, '{{}}'::jsonb) || {ph}").format(c=col_id, ph=pgsql.Placeholder())
            )
            params.append(Jsonb(merged_metadata))
        elif col in ("created_at", "updated_at"):
            select_parts.append(pgsql.SQL("NOW()"))
        else:
            select_parts.append(col_id)
    params.append(src_thread_id)

    stmt = pgsql.SQL("INSERT INTO {t} ({cols}) SELECT {vals} FROM {t} WHERE {tid} = {ph}").format(
        t=pgsql.Identifier("thread"),
        cols=pgsql.SQL(", ").join(pgsql.Identifier(c) for c in cols),
        vals=pgsql.SQL(", ").join(select_parts),
        tid=pgsql.Identifier("thread_id"),
        ph=pgsql.Placeholder(),
    )
    result = await conn.execute(stmt, params)
    if result.rowcount == 0:
        raise RuntimeError(f"Source thread {src_thread_id!r} disappeared before copy; rolling back")


async def copy_thread_atomically(
    *,
    src_thread_id: str,
    new_thread_id: str,
    user_identity: str,
    metadata_overrides: dict[str, Any] | None = None,
) -> None:
    """Copy thread row + checkpoint history in one REPEATABLE READ tx.

    The ``thread`` row and all checkpoint tables are read within the same
    pinned snapshot, so a concurrent writer cannot produce a partially
    copied result. See ``_copy_thread_row`` for the per-column semantics.
    """
    if db_manager.lg_pool is None:
        raise RuntimeError("Checkpoint pool is not initialized")

    async with db_manager.lg_pool.connection() as conn, conn.transaction():
        # First statement in tx — Postgres honours the level only before any
        # query runs. psycopg3 transaction() has no isolation_level kwarg and
        # setting conn.isolation_level would leak across pooled borrowers.
        await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        await _copy_thread_row(
            conn,
            src_thread_id=src_thread_id,
            new_thread_id=new_thread_id,
            user_identity=user_identity,
            metadata_overrides=metadata_overrides or {},
        )
        for table in _CHECKPOINT_TABLES:
            await _copy_checkpoint_table(conn, table, src_thread_id, new_thread_id)
    logger.info(
        "thread.copy",
        src_thread_id=src_thread_id,
        new_thread_id=new_thread_id,
        user_identity=user_identity,
    )
