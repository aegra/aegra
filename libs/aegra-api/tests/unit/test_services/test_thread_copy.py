"""Unit tests for thread_copy service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from psycopg import sql as pgsql

from aegra_api.services.thread_copy import (
    _CHECKPOINT_TABLES,
    _copy_checkpoint_table,
    _table_columns,
    copy_thread_atomically,
)


def _make_async_cursor(column_names: list[str]) -> AsyncMock:
    """Build an awaitable cursor whose ``fetchall`` returns ``dict_row``-style rows.

    Mirrors the pool's ``row_factory=dict_row`` configuration (see
    ``core/database.py``): each row is a dict keyed by column name.
    """
    cur = AsyncMock()
    cur.fetchall = AsyncMock(return_value=[{"column_name": name} for name in column_names])
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
        assert "information_schema.columns" in sql
        assert params == ("checkpoints",)


class TestCopyCheckpointTable:
    @pytest.mark.asyncio()
    async def test_builds_insert_select_with_thread_id_substitution(self) -> None:
        cols_cursor = _make_async_cursor(["thread_id", "checkpoint_ns", "checkpoint_id", "metadata"])
        insert_cursor = _make_async_cursor([])
        conn = _make_async_connection([cols_cursor, insert_cursor])

        await _copy_checkpoint_table(conn, "checkpoints", "src-uuid", "new-uuid")

        # Two execute calls: information_schema lookup + INSERT...SELECT
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
    async def test_inserts_thread_row_then_copies_three_checkpoint_tables(self) -> None:
        """The thread INSERT must precede the checkpoint INSERT...SELECT calls,
        all inside the same connection transaction."""
        mock_pool, mock_conn = _make_atomic_pool_mocks()

        with (
            patch("aegra_api.services.thread_copy._copy_checkpoint_table", new=AsyncMock()) as mock_copy,
            patch("aegra_api.services.thread_copy.db_manager") as mock_db,
        ):
            mock_db.lg_pool = mock_pool

            await copy_thread_atomically(
                src_thread_id="src-uuid",
                new_thread_id="new-uuid",
                src_status="idle",
                src_metadata={"k": "v"},
                user_identity="user-1",
            )

        # Thread INSERT issued exactly once on the same conn used for the
        # checkpoint copy (single transaction).
        assert mock_conn.execute.await_count == 1
        thread_sql, thread_params = mock_conn.execute.await_args_list[0].args
        assert thread_sql.startswith('INSERT INTO "thread"')
        assert thread_params[0] == "new-uuid"
        assert thread_params[1] == "idle"
        assert thread_params[3] == "user-1"
        # The checkpoint copy was invoked once per checkpoint table, in order.
        tables_copied = [call.args[1] for call in mock_copy.await_args_list]
        assert tables_copied == list(_CHECKPOINT_TABLES)
        for call in mock_copy.await_args_list:
            # Same connection ref ⇒ same Postgres tx as the thread INSERT.
            assert call.args[0] is mock_conn
            assert call.args[2] == "src-uuid"
            assert call.args[3] == "new-uuid"

    @pytest.mark.asyncio()
    async def test_serializes_metadata_to_json(self) -> None:
        """Metadata dict is JSON-serialized before being passed to the JSONB cast."""
        mock_pool, mock_conn = _make_atomic_pool_mocks()

        with (
            patch("aegra_api.services.thread_copy._copy_checkpoint_table", new=AsyncMock()),
            patch("aegra_api.services.thread_copy.db_manager") as mock_db,
        ):
            mock_db.lg_pool = mock_pool

            await copy_thread_atomically(
                src_thread_id="src",
                new_thread_id="new",
                src_status="idle",
                src_metadata={"owner": "alice", "graph_id": "agent"},
                user_identity="alice",
            )

        thread_params = mock_conn.execute.await_args_list[0].args[1]
        # metadata_json is the third positional after (new_id, status, json, ...)
        assert json.loads(thread_params[2]) == {"owner": "alice", "graph_id": "agent"}

    @pytest.mark.asyncio()
    async def test_raises_when_pool_not_initialized(self) -> None:
        with patch("aegra_api.services.thread_copy.db_manager") as mock_db:
            mock_db.lg_pool = None
            with pytest.raises(RuntimeError, match="not initialized"):
                await copy_thread_atomically(
                    src_thread_id="src",
                    new_thread_id="new",
                    src_status="idle",
                    src_metadata={},
                    user_identity="u",
                )
