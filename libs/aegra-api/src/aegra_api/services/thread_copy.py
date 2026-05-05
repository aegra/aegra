"""Thread copy service.

Implements deep-copy of a thread (its row in ``thread`` plus its checkpoint
history) at the SQL level so that ``checkpoint_id`` and the
``parent_checkpoint_id`` chain are preserved end-to-end, matching the
semantics of ``POST /threads/{id}/copy`` on LangSmith Deployments.

Both the thread row INSERT and the three checkpoint-table INSERT...SELECT
statements run inside a single Postgres transaction on the
``langgraph-checkpoint-postgres`` connection pool, so the copy is atomic:
either the new thread and its full checkpoint history land together, or
nothing does. Going through one pool also avoids cross-pool tx coordination
between SQLAlchemy and the checkpointer pool, which would otherwise leave
orphaned checkpoint rows on partial failures.

Column lists for the checkpoint tables are introspected at runtime via
``information_schema`` so the copy stays correct if the upstream
``langgraph-checkpoint-postgres`` schema gains columns (``task_path`` was
one such recent addition).
"""

import json
from typing import TYPE_CHECKING, Any

from psycopg import sql as pgsql

from aegra_api.core.database import db_manager

if TYPE_CHECKING:
    from psycopg import AsyncConnection

# Tables managed by langgraph-checkpoint-postgres. Each has ``thread_id``
# as a non-PK leading column; the copy preserves all other columns verbatim.
_CHECKPOINT_TABLES: tuple[str, ...] = ("checkpoints", "checkpoint_writes", "checkpoint_blobs")


async def _table_columns(conn: "AsyncConnection", table: str) -> list[str]:
    """Return the column names of ``table`` in declaration order.

    The aegra connection pool is configured with ``dict_row`` (see
    ``core/database.py`` — LangGraph requires dictionary rows), so each
    fetched row is a ``dict``. Access columns by name to stay compatible
    with the pool factory rather than relying on positional indexing.
    """
    cur = await conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s "
        "ORDER BY ordinal_position",
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
    """Copy all rows of ``table`` matching ``src_thread_id`` under ``new_thread_id``.

    Raises ``RuntimeError`` if the introspected schema lacks ``thread_id`` —
    a silent skip would let the surrounding transaction commit a partial
    copy (the new thread row plus the tables that did succeed), violating
    the atomicity guarantee. Raising lets ``conn.transaction()`` roll back
    the whole operation.

    Identifiers (table name + column names) are composed via
    ``psycopg.sql.Identifier`` rather than f-string quoting. The inputs
    come from ``information_schema``, so an exotic identifier containing a
    literal double-quote would silently break naive ``f'"{c}"'`` quoting;
    using the typed composer eliminates that class of bug.
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
    src_thread_id: str,
    new_thread_id: str,
    src_status: str,
    src_metadata: dict[str, Any],
    user_identity: str,
) -> None:
    """Atomically insert the new thread row and copy the checkpoint history.

    Runs the thread-row INSERT and the three checkpoint INSERT...SELECT
    statements inside one Postgres transaction. On any failure the whole
    copy is rolled back, preventing orphaned checkpoint rows or a thread
    row without its history.

    ``created_at`` and ``updated_at`` are set to ``NOW()`` to mark the time
    of the copy (matching LangSmith Deployments behaviour) rather than
    inheriting the source timestamps. Status and metadata are inherited
    from the source as-is — including statuses such as ``running`` or
    ``error`` if present.
    """
    if db_manager.lg_pool is None:
        raise RuntimeError("Checkpoint pool is not initialized")

    metadata_json = json.dumps(src_metadata or {})

    async with db_manager.lg_pool.connection() as conn, conn.transaction():
        await conn.execute(
            'INSERT INTO "thread" ("thread_id", "status", "metadata_json", "user_id", '
            '"created_at", "updated_at") '
            "VALUES (%s, %s, %s::jsonb, %s, NOW(), NOW())",
            (new_thread_id, src_status, metadata_json, user_identity),
        )
        for table in _CHECKPOINT_TABLES:
            await _copy_checkpoint_table(conn, table, src_thread_id, new_thread_id)
