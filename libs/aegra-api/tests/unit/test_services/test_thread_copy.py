"""Unit tests for thread_copy service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from psycopg import sql as pgsql
from psycopg.types.json import Jsonb

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
        verification against a live Postgres lives in the integration suite
        (see follow-up F3 in `infra/aegra/README.md`)."""
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
        """SET TRANSACTION → thread INSERT → checkpoint INSERT...SELECT calls,
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

        # Two execute calls on the connection: SET TRANSACTION first, then the
        # thread INSERT. Checkpoint tables go through the patched helper.
        assert mock_conn.execute.await_count == 2
        thread_sql, thread_params = mock_conn.execute.await_args_list[1].args
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
    async def test_sets_repeatable_read_isolation_before_dml(self) -> None:
        """``SET TRANSACTION ISOLATION LEVEL REPEATABLE READ`` must be the
        first statement inside the ``conn.transaction()`` block. PostgreSQL
        only honours the directive when no DML has run yet, so the order
        matters; without REPEATABLE READ the three checkpoint INSERT...SELECT
        statements can see different snapshots if a concurrent run on the
        source thread commits between them."""
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
                src_metadata={},
                user_identity="u",
            )

        first_sql = mock_conn.execute.await_args_list[0].args[0]
        assert "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ" in first_sql

    @pytest.mark.asyncio()
    async def test_serializes_metadata_via_jsonb_adapter(self) -> None:
        """Metadata is bound through psycopg's ``Jsonb`` adapter rather than
        a string + ``::jsonb`` cast. The adapter goes through psycopg's
        adapter chain, so non-JSON-serializable values (e.g. datetime, UUID)
        raise at adapt-time with a clear error rather than blowing up
        ``json.dumps`` mid-transaction."""
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
                src_metadata={"graph_id": "agent"},
                user_identity="alice",
            )

        thread_sql, thread_params = mock_conn.execute.await_args_list[1].args
        # No ``::jsonb`` cast in the SQL — psycopg infers the column type.
        assert "::jsonb" not in thread_sql
        # Third positional argument is the bound value: a Jsonb wrapper, not a
        # string, so psycopg adapts it natively.
        assert isinstance(thread_params[2], Jsonb)

    @pytest.mark.asyncio()
    async def test_metadata_owner_rewritten_to_caller(self) -> None:
        """``metadata.owner`` is overwritten with the caller identity. Without
        this rewrite the new thread inherits the source owner field, which
        misattributes the row for any code reading ``metadata.owner`` instead
        of the canonical ``thread.user_id`` column. ``create_thread`` enforces
        the same invariant for fresh threads."""
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
                user_identity="bob",
            )

        thread_params = mock_conn.execute.await_args_list[1].args[1]
        bound_metadata = thread_params[2].obj  # Jsonb stores the dict on `.obj`
        assert bound_metadata["owner"] == "bob"
        # Other keys are preserved verbatim.
        assert bound_metadata["graph_id"] == "agent"

    @pytest.mark.asyncio()
    async def test_does_not_mutate_caller_metadata_dict(self) -> None:
        """Owner rewrite must operate on a copy, otherwise the caller's
        ``ThreadORM.metadata_json`` dict (a SQLAlchemy mutable JSONB) would
        see ``owner`` change in place — a subtle ORM-state corruption."""
        mock_pool, _ = _make_atomic_pool_mocks()
        original = {"owner": "alice", "graph_id": "agent"}

        with (
            patch("aegra_api.services.thread_copy._copy_checkpoint_table", new=AsyncMock()),
            patch("aegra_api.services.thread_copy.db_manager") as mock_db,
        ):
            mock_db.lg_pool = mock_pool

            await copy_thread_atomically(
                src_thread_id="src",
                new_thread_id="new",
                src_status="idle",
                src_metadata=original,
                user_identity="bob",
            )

        assert original == {"owner": "alice", "graph_id": "agent"}

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
