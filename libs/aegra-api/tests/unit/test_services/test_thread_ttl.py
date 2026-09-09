"""Unit tests for thread TTL config resolution and the expiry sweep."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.postgres.base import MIGRATIONS, SELECT_SQL
from psycopg import Error as PsycopgError
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from aegra_api.observability.metrics import THREAD_TTL_SWEPT
from aegra_api.services.thread_ttl import (
    ThreadTTLConfig,
    ThreadTTLSweeper,
    _apply_strategy,
    _expired_claim_stmt,
    _process_expired_batch,
    get_thread_ttl_config,
    prune_expired_threads_for_user,
)
from aegra_api.settings import settings


def _swept_count(outcome: str) -> float:
    """Read the current value of the TTL counter for one outcome label."""
    return THREAD_TTL_SWEPT.labels(outcome=outcome)._value.get()


def _make_lg_pool() -> tuple[MagicMock, AsyncMock]:
    """Fake db_manager.lg_pool: connection() and transaction() async CMs.

    The prunable-history probe answers True by default; override
    conn.execute.return_value.fetchone for the skip path.
    """
    conn = AsyncMock()
    conn.execute.return_value.fetchone = AsyncMock(return_value={"has_prunable_history": True})
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    pool_cm = AsyncMock()
    pool_cm.__aenter__ = AsyncMock(return_value=conn)
    pool_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=pool_cm)
    return pool, conn


@pytest.fixture(autouse=True)
def _clear_ttl_config_cache() -> Iterator[None]:
    get_thread_ttl_config.cache_clear()
    yield
    get_thread_ttl_config.cache_clear()


@pytest.fixture(autouse=True)
def _no_env_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", None)
    monkeypatch.setattr(settings.thread_ttl, "LANGGRAPH_THREAD_TTL", None)


class TestResolveConfig:
    """Tests for get_thread_ttl_config source precedence and validation."""

    def test_returns_none_when_no_source_configured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        assert get_thread_ttl_config() is None

    def test_loads_from_aegra_json_checkpointer_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "checkpointer": {"ttl": {"strategy": "keep_latest", "default_ttl": 1440}},
                }
            )
        )

        config = get_thread_ttl_config()

        assert config is not None
        assert config.strategy == "keep_latest"
        assert config.default_ttl == 1440
        assert config.sweep_interval_minutes == 5
        assert config.sweep_limit == 10000

    def test_env_bare_number_sets_default_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "43200")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 43200
        assert config.strategy == "delete"

    def test_env_json_object_sets_all_fields(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            settings.thread_ttl,
            "AEGRA_THREAD_TTL",
            json.dumps(
                {
                    "strategy": "keep_latest",
                    "default_ttl": 60,
                    "sweep_interval_minutes": 1,
                    "sweep_limit": 50,
                }
            ),
        )

        config = get_thread_ttl_config()

        assert config is not None
        assert config.strategy == "keep_latest"
        assert config.default_ttl == 60
        assert config.sweep_interval_minutes == 1
        assert config.sweep_limit == 50

    def test_env_replaces_aegra_json_block_entirely(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whole-source override: json keys do not leak under an env config."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "checkpointer": {"ttl": {"strategy": "keep_latest", "sweep_limit": 7}},
                }
            )
        )
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "120")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 120
        assert config.strategy == "delete"
        assert config.sweep_limit == 10000

    def test_invalid_env_json_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "{not json")

        with pytest.raises(json.JSONDecodeError):
            get_thread_ttl_config()

    def test_invalid_strategy_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", json.dumps({"strategy": "purge"}))

        with pytest.raises(ValidationError):
            get_thread_ttl_config()

    def test_non_positive_default_ttl_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "0")

        with pytest.raises(ValidationError):
            get_thread_ttl_config()

    def test_langgraph_alias_used_as_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env files migrated from LangGraph Platform work unchanged."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "LANGGRAPH_THREAD_TTL", "777")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 777

    def test_aegra_var_wins_over_langgraph_alias(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "111")
        monkeypatch.setattr(settings.thread_ttl, "LANGGRAPH_THREAD_TTL", "777")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 111

    def test_blank_env_falls_back_to_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "checkpointer": {"ttl": {"default_ttl": 15}},
                }
            )
        )
        monkeypatch.setattr(settings.thread_ttl, "AEGRA_THREAD_TTL", "   ")

        config = get_thread_ttl_config()

        assert config is not None
        assert config.default_ttl == 15


class TestThreadTTLConfigModel:
    """Bounds validation on the config model itself."""

    def test_defaults(self) -> None:
        config = ThreadTTLConfig()

        assert config.strategy == "delete"
        assert config.default_ttl == 43200
        assert config.sweep_interval_minutes == 5
        assert config.sweep_limit == 10000

    @pytest.mark.parametrize(
        "field",
        ["default_ttl", "sweep_interval_minutes", "sweep_limit"],
    )
    def test_rejects_non_positive_values(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ThreadTTLConfig.model_validate({field: 0})

    @pytest.mark.parametrize("value", [float("inf"), 1e308])
    def test_rejects_timedelta_unsafe_ttl(self, value: float) -> None:
        """Values above MAX_TTL_MINUTES would overflow timedelta at insert time."""
        with pytest.raises(ValidationError):
            ThreadTTLConfig.model_validate({"default_ttl": value})


class TestClaimQuery:
    """The claim statement locks thread_ttl and thread rows and skips busy threads."""

    def test_sweep_claim_shape(self) -> None:
        stmt = _expired_claim_stmt(now=datetime.now(UTC), limit=10)
        sql = str(stmt.compile(dialect=postgresql.dialect()))

        assert "FOR UPDATE OF thread_ttl, thread SKIP LOCKED" in sql
        assert "EXISTS" in sql
        assert "expires_at <=" in sql
        assert "LIMIT" in sql
        assert "runs" in sql  # active-run guard subquery

    def test_exclude_ids_renders_not_in(self) -> None:
        stmt = _expired_claim_stmt(now=datetime.now(UTC), limit=10, exclude_ids={"failed-1"})
        sql = str(stmt.compile(dialect=postgresql.dialect()))

        assert "NOT IN" in sql

    def test_prune_claim_is_user_scoped(self) -> None:
        stmt = _expired_claim_stmt(now=datetime.now(UTC), limit=10, user_id="user-1")
        sql = str(stmt.compile(dialect=postgresql.dialect()))

        assert "thread.user_id" in sql
        assert "FOR UPDATE OF thread_ttl, thread SKIP LOCKED" in sql


class TestDeleteStrategy:
    """strategy=delete removes checkpoints strictly before the thread row."""

    @pytest.mark.asyncio
    async def test_deletes_checkpoints_before_thread_row(self) -> None:
        order: list[str] = []
        checkpointer = AsyncMock()
        checkpointer.adelete_thread.side_effect = lambda _tid: order.append("checkpoints")
        db = MagicMock()
        db.get_checkpointer.return_value = checkpointer
        session = AsyncMock()
        session.execute.side_effect = lambda _stmt: order.append("thread_row")

        with patch("aegra_api.services.thread_ttl.db_manager", db):
            outcome = await _apply_strategy(
                session, thread_id="t-1", strategy="delete", ttl_minutes=5.0, now=datetime.now(UTC)
            )

        assert outcome == "deleted"
        assert order == ["checkpoints", "thread_row"]
        checkpointer.adelete_thread.assert_awaited_once_with("t-1")


class TestKeepLatest:
    """strategy=keep_latest prunes history on the lg_pool and re-arms the TTL."""

    @pytest.mark.asyncio
    async def test_prunes_three_tables_in_order_and_rearms(self) -> None:
        pool, conn = _make_lg_pool()
        db = MagicMock()
        db.lg_pool = pool
        session = AsyncMock()

        with patch("aegra_api.services.thread_ttl.db_manager", db):
            outcome = await _apply_strategy(
                session, thread_id="t-1", strategy="keep_latest", ttl_minutes=30.0, now=datetime.now(UTC)
            )

        assert outcome == "pruned"
        statements = [call.args[0] for call in conn.execute.await_args_list]
        assert len(statements) == 4
        assert "SELECT EXISTS" in statements[0]  # prunable-history probe first
        assert "DELETE FROM checkpoints" in statements[1]
        assert "DELETE FROM checkpoint_writes" in statements[2]
        assert "DELETE FROM checkpoint_blobs" in statements[3]
        for call in conn.execute.await_args_list:
            assert "%(tid)s" in call.args[0]
            assert call.args[1] == {"tid": "t-1"}
        # Re-arm lands on the SQLAlchemy session, not the lg_pool
        rearm_sql = str(session.execute.await_args_list[0].args[0].compile(dialect=postgresql.dialect()))
        assert "UPDATE thread_ttl" in rearm_sql
        assert "expires_at" in rearm_sql

    @pytest.mark.asyncio
    async def test_skips_deletes_when_history_already_compact(self) -> None:
        """Idle expiry cycles cost one PK-indexed probe, not three DELETEs."""
        pool, conn = _make_lg_pool()
        conn.execute.return_value.fetchone = AsyncMock(return_value={"has_prunable_history": False})
        db = MagicMock()
        db.lg_pool = pool
        session = AsyncMock()

        with patch("aegra_api.services.thread_ttl.db_manager", db):
            outcome = await _apply_strategy(
                session, thread_id="t-1", strategy="keep_latest", ttl_minutes=30.0, now=datetime.now(UTC)
            )

        assert outcome == "pruned"
        assert conn.execute.await_count == 1  # probe only
        conn.transaction.assert_not_called()
        # Still re-arms so the next cycle stays scheduled
        rearm_sql = str(session.execute.await_args_list[0].args[0].compile(dialect=postgresql.dialect()))
        assert "UPDATE thread_ttl" in rearm_sql

    @pytest.mark.asyncio
    async def test_raises_when_db_not_initialized(self) -> None:
        db = MagicMock()
        db.lg_pool = None
        session = AsyncMock()

        with (
            patch("aegra_api.services.thread_ttl.db_manager", db),
            pytest.raises(RuntimeError, match="not initialized"),
        ):
            await _apply_strategy(
                session, thread_id="t-1", strategy="keep_latest", ttl_minutes=30.0, now=datetime.now(UTC)
            )


class TestFailureIsolation:
    """A checkpointer failure skips the item without poisoning the batch."""

    @pytest.mark.asyncio
    async def test_psycopg_error_skips_item_and_continues(self) -> None:
        checkpointer = AsyncMock()
        checkpointer.adelete_thread.side_effect = [PsycopgError("backend down"), None]
        db = MagicMock()
        db.get_checkpointer.return_value = checkpointer

        claim_result = MagicMock()
        claim_result.all.return_value = [("t-1", "delete", 5.0), ("t-2", "delete", 5.0)]
        session = AsyncMock()
        session.execute.side_effect = [claim_result, MagicMock()]

        errors_before = _swept_count("error")
        with patch("aegra_api.services.thread_ttl.db_manager", db):
            claimed, deleted, pruned, failed_ids = await _process_expired_batch(session, MagicMock(), datetime.now(UTC))

        assert (claimed, deleted, pruned) == (2, 1, 0)
        assert failed_ids == ["t-1"]
        assert checkpointer.adelete_thread.await_count == 2
        session.commit.assert_awaited_once()
        assert _swept_count("error") == errors_before + 1

    @pytest.mark.asyncio
    async def test_empty_claim_returns_zero_without_commit(self) -> None:
        claim_result = MagicMock()
        claim_result.all.return_value = []
        session = AsyncMock()
        session.execute.return_value = claim_result

        claimed, deleted, pruned, failed_ids = await _process_expired_batch(session, MagicMock(), datetime.now(UTC))

        assert (claimed, deleted, pruned, failed_ids) == (0, 0, 0, [])
        session.commit.assert_not_awaited()


class TestSweepLimit:
    """_tick claims sub-batches until sweep_limit is reached."""

    @pytest.mark.asyncio
    async def test_stops_at_sweep_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sweeper = ThreadTTLSweeper()
        config = ThreadTTLConfig(sweep_limit=250)
        monkeypatch.setattr("aegra_api.services.thread_ttl.get_thread_ttl_config", lambda: config)

        session = AsyncMock()
        maker = MagicMock()
        maker.return_value.__aenter__ = AsyncMock(return_value=session)
        maker.return_value.__aexit__ = AsyncMock(return_value=False)

        batches = [(100, 100, 0, []), (100, 100, 0, []), (50, 50, 0, []), (0, 0, 0, [])]
        with (
            patch("aegra_api.services.thread_ttl._get_session_maker", return_value=maker),
            patch(
                "aegra_api.services.thread_ttl._process_expired_batch",
                new_callable=AsyncMock,
                side_effect=batches,
            ) as mock_batch,
        ):
            await sweeper._tick()

        assert mock_batch.await_count == 3
        limits = [call.args[1]._limit_clause.value for call in mock_batch.await_args_list]
        assert limits == [100, 100, 50]

    @pytest.mark.asyncio
    async def test_failed_rows_excluded_from_subsequent_claims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Persistently failing rows sit first in expiry order; without the
        exclusion they'd be re-claimed every batch and starve later rows."""
        sweeper = ThreadTTLSweeper()
        config = ThreadTTLConfig(sweep_limit=1000)
        monkeypatch.setattr("aegra_api.services.thread_ttl.get_thread_ttl_config", lambda: config)

        session = AsyncMock()
        maker = MagicMock()
        maker.return_value.__aenter__ = AsyncMock(return_value=session)
        maker.return_value.__aexit__ = AsyncMock(return_value=False)

        batches = [(2, 1, 0, ["bad-1"]), (1, 1, 0, []), (0, 0, 0, [])]
        with (
            patch("aegra_api.services.thread_ttl._get_session_maker", return_value=maker),
            patch(
                "aegra_api.services.thread_ttl._process_expired_batch",
                new_callable=AsyncMock,
                side_effect=batches,
            ) as mock_batch,
        ):
            await sweeper._tick()

        assert mock_batch.await_count == 3
        second_claim = str(mock_batch.await_args_list[1].args[1].compile(dialect=postgresql.dialect()))
        assert "NOT IN" in second_claim

    @pytest.mark.asyncio
    async def test_tick_is_noop_without_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sweeper = ThreadTTLSweeper()
        monkeypatch.setattr("aegra_api.services.thread_ttl.get_thread_ttl_config", lambda: None)

        with patch("aegra_api.services.thread_ttl._process_expired_batch", new_callable=AsyncMock) as mock_batch:
            await sweeper._tick()

        mock_batch.assert_not_awaited()


class TestStartStop:
    """Sweeper lifecycle mirrors the cron scheduler."""

    @pytest.mark.asyncio
    async def test_loop_ticks_first_then_sleeps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sweeper = ThreadTTLSweeper()
        config = ThreadTTLConfig(sweep_interval_minutes=0.001)
        monkeypatch.setattr("aegra_api.services.thread_ttl.get_thread_ttl_config", lambda: config)

        tick_count = 0

        async def counting_tick() -> None:
            nonlocal tick_count
            tick_count += 1
            sweeper._running = False

        sweeper._running = True
        with patch.object(sweeper, "_tick", side_effect=counting_tick):
            await sweeper._loop()

        assert tick_count == 1

    @pytest.mark.asyncio
    async def test_loop_survives_tick_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sweeper = ThreadTTLSweeper()
        config = ThreadTTLConfig(sweep_interval_minutes=0.001)
        monkeypatch.setattr("aegra_api.services.thread_ttl.get_thread_ttl_config", lambda: config)

        call_count = 0

        async def failing_tick() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("boom")
            sweeper._running = False

        sweeper._running = True
        with patch.object(sweeper, "_tick", side_effect=failing_tick):
            await sweeper._loop()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_start_creates_task_and_stop_cancels_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sweeper = ThreadTTLSweeper()
        config = ThreadTTLConfig(sweep_interval_minutes=60)
        monkeypatch.setattr("aegra_api.services.thread_ttl.get_thread_ttl_config", lambda: config)

        with patch.object(sweeper, "_tick", new_callable=AsyncMock):
            await sweeper.start()
            task = sweeper._task
            assert task is not None
            assert not task.done()

            await sweeper.stop()

        assert sweeper._task is None
        assert task.done()


class TestSchemaCanary:
    """Fail loudly if a langgraph-checkpoint-postgres bump changes the schema
    the keep_latest pruning SQL is pinned to."""

    def test_checkpoint_tables_match_pruning_assumptions(self) -> None:
        ddl = "\n".join(MIGRATIONS)
        assert "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)" in ddl
        assert "PRIMARY KEY (thread_id, checkpoint_ns, channel, version)" in ddl
        assert "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)" in ddl
        # keep_latest keeps exactly the blob versions the latest checkpoint
        # references — the same join langgraph's reader performs.
        assert "jsonb_each_text(checkpoint -> 'channel_versions')" in SELECT_SQL


class TestPruneForUser:
    """prune_expired_threads_for_user loops user-scoped claims until dry."""

    @pytest.mark.asyncio
    async def test_accumulates_counts_until_no_more_claims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("aegra_api.services.thread_ttl.get_thread_ttl_config", lambda: None)
        session = AsyncMock()

        batches = [(2, 1, 1, ["bad-1"]), (1, 1, 0, []), (0, 0, 0, [])]
        with patch(
            "aegra_api.services.thread_ttl._process_expired_batch",
            new_callable=AsyncMock,
            side_effect=batches,
        ) as mock_batch:
            deleted, pruned = await prune_expired_threads_for_user(session, user_id="user-1")

        assert (deleted, pruned) == (2, 1)
        assert mock_batch.await_count == 3
        claim_sql = str(mock_batch.await_args_list[0].args[1].compile(dialect=postgresql.dialect()))
        assert "thread.user_id" in claim_sql
        assert "FOR UPDATE OF thread_ttl, thread SKIP LOCKED" in claim_sql
