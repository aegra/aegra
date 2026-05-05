"""Thread copy service.

Implements deep-copy of a thread (its row in ``thread`` plus its checkpoint
history) at the SQL level so that ``checkpoint_id`` and the
``parent_checkpoint_id`` chain are preserved end-to-end, matching the
semantics of ``POST /threads/{id}/copy`` on LangSmith Deployments.

Both the thread row INSERT and the three checkpoint-table INSERT...SELECT
statements run inside a single Postgres transaction on the
``langgraph-checkpoint-postgres`` connection pool, raised to
REPEATABLE READ isolation. Without that level, a concurrent writer on the
source thread (e.g. a run mid-execution) could commit between the three
INSERT...SELECT statements, leaving the new thread with an inconsistent
slice of the checkpoint history. REPEATABLE READ pins the snapshot at
transaction start so the three SELECTs see the same consistent view, and
on any failure the whole copy is rolled back.

Going through one pool also avoids cross-pool tx coordination between
SQLAlchemy and the checkpointer pool, which would otherwise leave
orphaned checkpoint rows on partial failures.

Column lists for the checkpoint tables are introspected at runtime via
``pg_catalog.pg_attribute`` keyed on ``to_regclass(<table>)``. This is
the same path the unqualified ``INSERT INTO <table>`` statements use to
resolve their target relation — they both follow the connection's
``search_path``. Using ``information_schema.columns`` filtered by
``current_schema()`` would only inspect the *first* schema in the
search path, which is fine for default deployments but breaks under
non-default ``search_path`` configurations (the introspection returns
zero rows even though the unqualified DML still finds the table in a
later schema). Forward-compat: new columns added upstream
(``task_path`` was one such recent addition) are picked up automatically.
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
    """Return the column names of ``table`` in declaration order.

    Resolves ``table`` through ``to_regclass`` so the introspection
    follows the connection's ``search_path`` exactly the way an
    unqualified ``INSERT INTO <table>`` would. Filters out dropped
    columns via ``attisdropped`` — ``pg_attribute`` retains tombstone
    rows for dropped columns, unlike ``information_schema.columns``.

    The aegra connection pool is configured with ``dict_row`` (see
    ``core/database.py`` — LangGraph requires dictionary rows), so each
    fetched row is a ``dict``. The ``AS column_name`` alias preserves
    that contract for callers (and existing tests).
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
    """Copy all rows of ``table`` matching ``src_thread_id`` under ``new_thread_id``.

    Raises ``RuntimeError`` if the introspected schema lacks ``thread_id`` —
    a silent skip would let the surrounding transaction commit a partial
    copy (the new thread row plus the tables that did succeed), violating
    the atomicity guarantee. Raising lets ``conn.transaction()`` roll back
    the whole operation.

    Identifiers (table name + column names) are composed via
    ``psycopg.sql.Identifier`` rather than f-string quoting. The inputs
    come from ``pg_attribute``, so an exotic identifier containing a
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
    statements inside one Postgres transaction at REPEATABLE READ
    isolation. On any failure the whole copy is rolled back, preventing
    orphaned checkpoint rows or a thread row without its history.

    The new thread's ``metadata.owner`` is rewritten to ``user_identity``
    (mirroring ``create_thread``, which enforces the same invariant for
    fresh threads). Without this, the source thread's owner attribution
    would be inherited by the copy — an internal-consistency bug for any
    code that reads ``metadata.owner`` instead of the canonical
    ``thread.user_id`` column. ``user_id`` is correctly bound to the
    caller via the dedicated parameter.

    ``created_at`` and ``updated_at`` are set to ``NOW()`` to mark the
    time of the copy (matching LangSmith Deployments behaviour) rather
    than inheriting the source timestamps. Status is inherited from the
    source as-is — including statuses such as ``running`` or ``error``
    if present.
    """
    if db_manager.lg_pool is None:
        raise RuntimeError("Checkpoint pool is not initialized")

    # Shallow copy is sufficient — we only mutate the top-level ``owner``
    # key, so any nested values shared with the caller's dict are safe to
    # alias.  ``dict(None or {})`` also defends against a ``None`` argument
    # without an explicit branch.
    metadata = dict(src_metadata or {})
    metadata["owner"] = user_identity

    async with db_manager.lg_pool.connection() as conn, conn.transaction():
        # Pin the transaction snapshot so the three checkpoint INSERT...SELECT
        # statements see a consistent view of the source thread, even if a
        # concurrent run on that thread is committing checkpoints. Must be
        # the first statement in the transaction for PostgreSQL to honour it.
        await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        await conn.execute(
            'INSERT INTO "thread" ("thread_id", "status", "metadata_json", "user_id", '
            '"created_at", "updated_at") '
            "VALUES (%s, %s, %s, %s, NOW(), NOW())",
            (new_thread_id, src_status, Jsonb(metadata), user_identity),
        )
        for table in _CHECKPOINT_TABLES:
            await _copy_checkpoint_table(conn, table, src_thread_id, new_thread_id)
    # Audit log on success — ISO 27017/27018 expect data-duplication events
    # to be traceable to caller, source, target, and timestamp. The failure
    # path emits ``logger.exception`` from the API layer; this completes the
    # pair so the operation is observable in both outcomes.
    logger.info(
        "thread.copy",
        src_thread_id=src_thread_id,
        new_thread_id=new_thread_id,
        user_identity=user_identity,
    )
