"""Real PostgreSQL regression for advisory locks and concurrent index builds."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from aegra_api.core import migrations


def test_waiting_migration_does_not_block_concurrent_index_build(monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = os.environ.get("AEGRA_MIGRATION_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set AEGRA_MIGRATION_TEST_DATABASE_URL to an isolated PostgreSQL test database")

    suffix = uuid4().hex
    application_name = f"aegra_migration_test_{suffix}"
    table_name = sql.Identifier(f"migration_lock_{suffix}")
    index_name = f"migration_lock_idx_{suffix}"
    lock_url = make_conninfo(database_url, application_name=application_name)
    monkeypatch.setattr(migrations.settings, "db", SimpleNamespace(database_url_sync=lock_url))
    holder_ready = Event()
    start_index_build = Event()

    def build_index() -> None:
        with migrations.migration_advisory_lock(), psycopg.connect(database_url, autocommit=True) as connection:
            holder_ready.set()
            assert start_index_build.wait(timeout=10), "contending migration never connected"
            connection.execute("SET statement_timeout = '3s'")
            connection.execute(
                sql.SQL("CREATE INDEX CONCURRENTLY {} ON {} USING gin (metadata)").format(
                    sql.Identifier(index_name), table_name
                )
            )

    def waiting_migration() -> None:
        with migrations.migration_advisory_lock():
            pass

    with psycopg.connect(database_url, autocommit=True) as observer:
        observer.execute(sql.SQL("CREATE TABLE {} (metadata jsonb NOT NULL)").format(table_name))
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                holder = executor.submit(build_index)
                assert holder_ready.wait(timeout=10), "first migration failed to acquire its lock"
                contender = executor.submit(waiting_migration)
                deadline = time.monotonic() + 5
                try:
                    while time.monotonic() < deadline:
                        attempts = observer.execute(
                            "SELECT state, wait_event, backend_xmin, query "
                            "FROM pg_stat_activity WHERE application_name = %s",
                            (application_name,),
                        ).fetchall()
                        blocking_wait = any(row[1] == "advisory" for row in attempts)
                        polling_wait = all(
                            row[0] == "idle" and row[2] is None and "pg_try_advisory_lock" in row[3] for row in attempts
                        )
                        if len(attempts) == 2 and (blocking_wait or polling_wait):
                            break
                        time.sleep(0.01)
                    else:
                        pytest.fail("second migration failed to open its lock connection")
                finally:
                    start_index_build.set()
                holder.result(timeout=10)
                contender.result(timeout=10)

            valid = observer.execute(
                "SELECT indisvalid FROM pg_index WHERE indexrelid = %s::regclass", (index_name,)
            ).fetchone()
            assert valid == (True,)
        finally:
            observer.execute(sql.SQL("DROP TABLE {} CASCADE").format(table_name))
