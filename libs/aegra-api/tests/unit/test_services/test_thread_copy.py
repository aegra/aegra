"""Unit tests for thread_copy service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from psycopg import sql as pgsql
from psycopg.types.json import Jsonb

from aegra_api.services.thread_copy import (
    _CHECKPOINT_TABLES,
    _copy_checkpoint_table,
    _copy_thread_row,
    _table_columns,
    copy_thread_atomically,
)

_THREAD_COLS: list[str] = ["thread_id", "status", "metadata_json", "user_id", "created_at", "updated_at"]


def _make_async_cursor(column_names: list[str], rowcount: int = 1) -> AsyncMock:
    """Build an awaitable cursor whose ``fetchall`` returns ``dict_row``-style rows.

    Mirrors the pool's ``row_factory=dict_row`` configuration (see
    ``core/database.py``): each row is a dict keyed by column name.
    ``rowcount`` backs the affected-row check after an ``INSERT``.
    """
    cur = AsyncMock()
    cur.fetchall = AsyncMock(return_value=[{"column_name": name} for name in column_names])
    cur.rowcount = rowcount
    return cur


def _make_async_connection(cursors: list[AsyncMock]) -> AsyncMock:
    """Build an async connection whose ``execute`` yields ``cursors`` in order."""
    conn = AsyncMock()
    conn.execute = AsyncMock(side_effect=cursors)
    return conn


class TestTableColumns:
    @pytest.mark.asyncio()
    async def test_returns_columns_in_declaration_order(self) -> None:
        cursor = _make_async_cursor(["thread_id", "checkpoint_id", "metadata"])
        conn = _make_async_connection([cursor])

        result = await _table_columns(conn, "checkpoints")

        assert result == ["thread_id", "checkpoint_id", "metadata"]
        # Query is parametrised on table name — no string interpolation.
        args, _ = conn.execute.call_args
        sql, params = args
        # Introspection runs through ``to_regclass`` against ``pg_attribute``
        # so it follows the connection's ``search_path`` exactly the way an
        # unqualified ``INSERT INTO checkpoints`` does. Filtering on
        # ``current_schema()`` would only inspect the first schema in the
        # search path and silently miss tables that live in a later schema.
        assert "pg_catalog.pg_attribute" in sql
        assert "to_regclass" in sql
        assert "attisdropped" in sql  # tombstones excluded
        assert params == ("checkpoints",)


class TestCopyCheckpointTable:
    @pytest.mark.asyncio()
    async def test_builds_insert_select_with_thread_id_substitution(self) -> None:
        cols_cursor = _make_async_cursor(["thread_id", "checkpoint_ns", "checkpoint_id", "metadata"])
        insert_cursor = _make_async_cursor([])
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_checkpoint_table(conn, "checkpoints", "src-uuid", "new-uuid")

        # Two execute calls: pg_attribute introspection + INSERT...SELECT
        assert conn.execute.await_count == 2
        insert_call = conn.execute.await_args_list[1]
        stmt, params = insert_call.args
        # The statement is composed via psycopg.sql.Identifier rather than
        # an f-string, so render it once for content assertions.
        assert isinstance(stmt, pgsql.Composed)
        sql_str = stmt.as_string(None)
        assert sql_str.startswith('INSERT INTO "checkpoints"')
        assert '"thread_id"' in sql_str
        assert '"checkpoint_ns"' in sql_str
        assert '"checkpoint_id"' in sql_str
        assert '"metadata"' in sql_str
        assert "WHERE thread_id = %s" in sql_str
        # First param is the new thread_id, second is the source — order matters
        assert params == ("new-uuid", "src-uuid")

    @pytest.mark.asyncio()
    async def test_raises_when_thread_id_column_missing(self) -> None:
        """Unexpected schema (no ``thread_id`` col) raises so the surrounding
        transaction rolls back rather than committing a partial copy."""
        cols_cursor = _make_async_cursor(["checkpoint_id"])
        conn = _make_async_connection([cols_cursor])

        with pytest.raises(RuntimeError, match="thread_id"):
            await _copy_checkpoint_table(conn, "checkpoints", "src", "new")

        # Only the introspection call — no INSERT issued before the raise.
        assert conn.execute.await_count == 1

    @pytest.mark.asyncio()
    async def test_does_not_select_thread_id_twice(self) -> None:
        """thread_id must appear once in column list — first param of INSERT."""
        cols_cursor = _make_async_cursor(["thread_id", "checkpoint_id"])
        insert_cursor = _make_async_cursor([])
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_checkpoint_table(conn, "checkpoints", "src", "new")

        stmt = conn.execute.await_args_list[1].args[0]
        sql_str = stmt.as_string(None)
        # Non-thread_id column listed once after explicit thread_id literal.
        assert sql_str.count('"checkpoint_id"') == 2  # in INSERT cols + SELECT cols
        assert sql_str.count('"thread_id"') == 1  # only in INSERT cols

    @pytest.mark.asyncio()
    async def test_preserves_checkpoint_chain_columns_verbatim(self) -> None:
        """Both ``checkpoint_id`` and ``parent_checkpoint_id`` must appear in
        the SELECT clause unchanged, so the new thread inherits the source's
        chain identifiers byte-for-byte. This is the headline differentiator
        vs ``checkpoint-fork`` (which regenerates checkpoint IDs); end-to-end
        verification against a live Postgres lives in the e2e suite."""
        cols_cursor = _make_async_cursor(
            [
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "parent_checkpoint_id",
                "type",
                "checkpoint",
                "metadata",
            ]
        )
        insert_cursor = _make_async_cursor([])
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_checkpoint_table(conn, "checkpoints", "src", "new")

        stmt = conn.execute.await_args_list[1].args[0]
        sql_str = stmt.as_string(None)
        # checkpoint_id and parent_checkpoint_id appear in BOTH the INSERT
        # column list and the SELECT projection — meaning the value is read
        # from the source row and inserted verbatim under the new thread_id.
        # No transformation, no nextval(), no CASE WHEN.
        assert sql_str.count('"checkpoint_id"') == 2
        assert sql_str.count('"parent_checkpoint_id"') == 2
        # No id-regeneration artefacts in the SQL.
        assert "nextval" not in sql_str.lower()
        assert "uuid_generate" not in sql_str.lower()
        assert "gen_random_uuid" not in sql_str.lower()


class TestCopyThreadRow:
    @pytest.mark.asyncio()
    async def test_builds_insert_select_from_source_thread(self) -> None:
        cols_cursor = _make_async_cursor(_THREAD_COLS)
        insert_cursor = _make_async_cursor([], rowcount=1)
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_thread_row(conn, src_thread_id="src", new_thread_id="new", user_identity="u", metadata_overrides={})

        assert conn.execute.await_count == 2  # introspection + INSERT...SELECT
        stmt = conn.execute.await_args_list[1].args[0]
        sql_str = stmt.as_string(None)
        assert sql_str.startswith('INSERT INTO "thread"')
        assert 'FROM "thread" WHERE "thread_id" = %s' in sql_str
        assert "NOW()" in sql_str  # timestamps regenerated, not inherited

    @pytest.mark.asyncio()
    async def test_active_status_resets_to_idle(self) -> None:
        """Regression: a ``busy``/``running`` source must not copy into a
        permanently-stuck thread. The copy has no ``runs`` row to advance an
        active status, so the SQL normalises it to ``idle`` while leaving
        terminal states untouched."""
        cols_cursor = _make_async_cursor(_THREAD_COLS)
        insert_cursor = _make_async_cursor([], rowcount=1)
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_thread_row(conn, src_thread_id="src", new_thread_id="new", user_identity="u", metadata_overrides={})

        sql_str = conn.execute.await_args_list[1].args[0].as_string(None)
        assert "CASE WHEN \"status\" IN ('busy', 'running') THEN 'idle' ELSE \"status\" END" in sql_str

    @pytest.mark.asyncio()
    async def test_metadata_merges_source_then_handler_then_owner(self) -> None:
        """Merge order is source ``||`` handler overrides ``||`` ``owner``.
        The owner rewrite wins even over a handler-supplied ``owner`` key —
        it is the security invariant that attributes the copy to the caller."""
        cols_cursor = _make_async_cursor(_THREAD_COLS)
        insert_cursor = _make_async_cursor([], rowcount=1)
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_thread_row(
            conn,
            src_thread_id="src",
            new_thread_id="new",
            user_identity="bob",
            metadata_overrides={"team": "x", "owner": "alice"},
        )

        stmt, params = conn.execute.await_args_list[1].args
        sql_str = stmt.as_string(None)
        # Source metadata inherited via jsonb concat, handler/owner layered on top.
        assert "COALESCE(\"metadata_json\", '{}'::jsonb) ||" in sql_str
        jsonb_param = next(p for p in params if isinstance(p, Jsonb))
        assert jsonb_param.obj == {"team": "x", "owner": "bob"}

    @pytest.mark.asyncio()
    async def test_binds_identity_columns_as_params_in_column_order(self) -> None:
        cols_cursor = _make_async_cursor(_THREAD_COLS)
        insert_cursor = _make_async_cursor([], rowcount=1)
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_thread_row(
            conn, src_thread_id="src-uuid", new_thread_id="new-uuid", user_identity="user-1", metadata_overrides={}
        )

        params = conn.execute.await_args_list[1].args[1]
        # SELECT-list order (thread_id, metadata_json, user_id) then WHERE src.
        assert params[0] == "new-uuid"
        assert isinstance(params[1], Jsonb)
        assert params[2] == "user-1"
        assert params[-1] == "src-uuid"

    @pytest.mark.asyncio()
    async def test_inherits_unknown_future_column_verbatim(self) -> None:
        """Regression: a column outside the override set is copied as-is, so a
        NOT NULL column added upstream is inherited from the source rather than
        silently dropped from the INSERT."""
        cols = [*_THREAD_COLS, "tenant_id"]
        cols_cursor = _make_async_cursor(cols)
        insert_cursor = _make_async_cursor([], rowcount=1)
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_thread_row(conn, src_thread_id="src", new_thread_id="new", user_identity="u", metadata_overrides={})

        sql_str = conn.execute.await_args_list[1].args[0].as_string(None)
        # tenant_id in both the INSERT column list and the SELECT projection.
        assert sql_str.count('"tenant_id"') == 2

    @pytest.mark.asyncio()
    async def test_raises_when_thread_id_column_missing(self) -> None:
        cols_cursor = _make_async_cursor(["status", "user_id"])
        conn = _make_async_connection([cols_cursor])

        with pytest.raises(RuntimeError, match="thread_id"):
            await _copy_thread_row(
                conn, src_thread_id="src", new_thread_id="new", user_identity="u", metadata_overrides={}
            )

        assert conn.execute.await_count == 1  # introspection only, no INSERT

    @pytest.mark.asyncio()
    async def test_raises_when_source_row_vanished(self) -> None:
        """TOCTOU: the source may be deleted between the ownership check and
        the copy. Zero affected rows raises so the transaction rolls back
        instead of leaving a thread with no checkpoint history."""
        cols_cursor = _make_async_cursor(_THREAD_COLS)
        insert_cursor = _make_async_cursor([], rowcount=0)
        conn = _make_async_connection([cols_cursor, insert_cursor])

        with pytest.raises(RuntimeError, match="disappeared"):
            await _copy_thread_row(
                conn, src_thread_id="gone", new_thread_id="new", user_identity="u", metadata_overrides={}
            )


def _make_atomic_pool_mocks() -> tuple[MagicMock, MagicMock]:
    """Build a (pool, conn) pair wired with the async context-manager protocol."""
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()

    transaction_ctx = MagicMock()
    transaction_ctx.__aenter__ = AsyncMock(return_value=None)
    transaction_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=transaction_ctx)

    connection_ctx = MagicMock()
    connection_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    connection_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=connection_ctx)
    return mock_pool, mock_conn


class TestCopyThreadAtomically:
    @pytest.mark.asyncio()
    async def test_sets_repeatable_read_then_copies_thread_then_checkpoints(self) -> None:
        """SET TRANSACTION is the first statement, then the thread row, then
        each checkpoint table in order — all on the same connection so they
        share one Postgres transaction."""
        mock_pool, mock_conn = _make_atomic_pool_mocks()

        with (
            patch("aegra_api.services.thread_copy._copy_thread_row", new=AsyncMock()) as mock_thread,
            patch("aegra_api.services.thread_copy._copy_checkpoint_table", new=AsyncMock()) as mock_copy,
            patch("aegra_api.services.thread_copy.db_manager") as mock_db,
        ):
            mock_db.lg_pool = mock_pool

            await copy_thread_atomically(
                src_thread_id="src-uuid",
                new_thread_id="new-uuid",
                user_identity="user-1",
                metadata_overrides={"k": "v"},
            )

        # Only SET TRANSACTION runs directly on the connection; the row and
        # checkpoint copies go through the patched helpers.
        assert mock_conn.execute.await_count == 1
        assert "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ" in mock_conn.execute.await_args_list[0].args[0]

        mock_thread.assert_awaited_once()
        assert mock_thread.await_args.args[0] is mock_conn
        assert mock_thread.await_args.kwargs["src_thread_id"] == "src-uuid"
        assert mock_thread.await_args.kwargs["metadata_overrides"] == {"k": "v"}

        tables_copied = [call.args[1] for call in mock_copy.await_args_list]
        assert tables_copied == list(_CHECKPOINT_TABLES)
        for call in mock_copy.await_args_list:
            assert call.args[0] is mock_conn  # same connection ⇒ same tx

    @pytest.mark.asyncio()
    async def test_defaults_metadata_overrides_to_empty(self) -> None:
        mock_pool, _ = _make_atomic_pool_mocks()

        with (
            patch("aegra_api.services.thread_copy._copy_thread_row", new=AsyncMock()) as mock_thread,
            patch("aegra_api.services.thread_copy._copy_checkpoint_table", new=AsyncMock()),
            patch("aegra_api.services.thread_copy.db_manager") as mock_db,
        ):
            mock_db.lg_pool = mock_pool

            await copy_thread_atomically(src_thread_id="src", new_thread_id="new", user_identity="u")

        assert mock_thread.await_args.kwargs["metadata_overrides"] == {}

    @pytest.mark.asyncio()
    async def test_raises_when_pool_not_initialized(self) -> None:
        with patch("aegra_api.services.thread_copy.db_manager") as mock_db:
            mock_db.lg_pool = None
            with pytest.raises(RuntimeError, match="not initialized"):
                await copy_thread_atomically(src_thread_id="src", new_thread_id="new", user_identity="u")
