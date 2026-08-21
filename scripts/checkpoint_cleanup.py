#!/usr/bin/env python3
"""
LangGraph cleanup job (schema-aware).

This script enforces two retention policies in one pass.

Checkpoint retention:
  - Threads with thread.status='busy' or any in-flight run -> SKIPPED.
  - Threads idle/error AND thread.updated_at >= now() - 7 days -> keep all.
  - Threads idle/error AND thread.updated_at < now() - 7 days  -> keep only the
    latest checkpoint per (thread_id, checkpoint_ns), and only the blob
    versions referenced by that kept checkpoint's channel_versions map.
  - Thread rows themselves are NEVER deleted by this script.

Runs retention:
  - Runs in terminal statuses older than 30 days -> DELETE.
  - Runs with active leases (lease_expires_at >= now()) -> SKIPPED.
  - Runs in in-flight statuses -> SKIPPED.

Safe to run repeatedly. Idempotent. Batched. Logs everything.

Usage:
    python checkpoint_cleanup.py --dry-run
    python checkpoint_cleanup.py
    python checkpoint_cleanup.py --skip-runs
    python checkpoint_cleanup.py --skip-checkpoints
    python checkpoint_cleanup.py --skip-estimate            # skip slow count queries
    python checkpoint_cleanup.py --thread-id <id> --dry-run
    python checkpoint_cleanup.py --retention-days 14 --runs-retention-days 60

Requires:
    pip install psycopg[binary]>=3.1

Environment:
    DATABASE_URL  postgres://user:pass@host:port/db
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import suppress
from typing import Any

import psycopg  # type: ignore[import-not-found]
from psycopg import sql  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("checkpoint_cleanup")


def require_row(row: Any, *, context: str) -> tuple[Any, ...]:
    if row is None:
        raise RuntimeError(f"Expected a row for {context}, but query returned none")
    return row


# ---------------------------------------------------------------------------
# Status semantics (locked in based on your data)
# ---------------------------------------------------------------------------
THREAD_BUSY_STATUSES = ("busy",)
RUN_IN_FLIGHT_STATUSES = ("pending", "running")
RUN_TERMINAL_STATUSES = ("success", "error", "interrupted", "timeout", "cancelled")


# ---------------------------------------------------------------------------
# Cleanup steps - checkpoints
# ---------------------------------------------------------------------------
def build_checkpoint_working_sets(
    conn: psycopg.Connection,
    retention_days: int,
    thread_id: str | None = None,
    force_thread: bool = False,
) -> tuple[int, int]:
    """Populate temp tables identifying checkpoint cleanup work.

    Joins against `thread` and `runs` to enforce safety gates:
      - Skip thread.status='busy'
      - Skip threads with any in-flight run

    Returns (threads_to_trim_count, blobs_to_keep_count).
    """
    where_clauses: list[sql.SQL] = [
        sql.SQL("t.status <> ALL(%s)"),
        sql.SQL("NOT EXISTS (SELECT 1 FROM runs r WHERE r.thread_id = t.thread_id AND r.status = ANY(%s))"),
    ]
    params: list = [list(THREAD_BUSY_STATUSES), list(RUN_IN_FLIGHT_STATUSES)]

    if not force_thread:
        where_clauses.append(sql.SQL("t.updated_at < now() - make_interval(days => %s)"))
        params.append(retention_days)

    if thread_id:
        where_clauses.append(sql.SQL("t.thread_id = %s"))
        params.append(thread_id)

    where_sql = sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_clauses)
    threads_to_trim_sql = sql.SQL(
        """
        CREATE TEMP TABLE threads_to_trim AS
        SELECT c.thread_id, c.checkpoint_ns, MAX(c.checkpoint_id) AS latest_id
        FROM checkpoints c
        JOIN thread t ON t.thread_id = c.thread_id
        {}
        GROUP BY c.thread_id, c.checkpoint_ns
        """
    ).format(where_sql)

    with conn.cursor() as cur:
        log.info("Building threads_to_trim%s", f" (scoped to thread_id={thread_id})" if thread_id else "")
        if force_thread:
            log.warning("  --force-thread is ON: ignoring age cutoff for this thread")

        cur.execute("DROP TABLE IF EXISTS threads_to_trim")
        cur.execute(threads_to_trim_sql, params)
        cur.execute("CREATE INDEX ON threads_to_trim (thread_id, checkpoint_ns)")
        cur.execute("ANALYZE threads_to_trim")
        cur.execute("SELECT count(*) FROM threads_to_trim")
        threads_count = int(require_row(cur.fetchone(), context="threads_to_trim count")[0])
        log.info("  threads_to_trim: %s rows", f"{threads_count:,}")

        if threads_count == 0:
            return 0, 0

        log.info("Building blobs_to_keep")
        cur.execute("DROP TABLE IF EXISTS blobs_to_keep")
        cur.execute("""
            CREATE TEMP TABLE blobs_to_keep AS
            SELECT
                c.thread_id,
                c.checkpoint_ns,
                kv.key             AS channel,
                kv.value #>> '{}'  AS version
            FROM checkpoints c
            JOIN threads_to_trim t
              ON t.thread_id = c.thread_id
             AND t.checkpoint_ns = c.checkpoint_ns
             AND t.latest_id = c.checkpoint_id
            CROSS JOIN LATERAL jsonb_each(
                COALESCE(c.checkpoint -> 'channel_versions', '{}'::jsonb)
            ) kv
        """)
        cur.execute("CREATE INDEX ON blobs_to_keep (thread_id, checkpoint_ns, channel, version)")
        cur.execute("ANALYZE blobs_to_keep")
        cur.execute("SELECT count(*) FROM blobs_to_keep")
        blobs_count = int(require_row(cur.fetchone(), context="blobs_to_keep count")[0])
        log.info("  blobs_to_keep: %s rows", f"{blobs_count:,}")

        return threads_count, blobs_count


def estimate_checkpoint_deletes(
    conn: psycopg.Connection,
    retention_days: int,
    force_thread: bool,
) -> dict[str, int]:
    """Count rows that would be deleted, applying the same per-batch
    revalidation gates as the DELETE statements. So the dry-run estimate
    matches actual deletion behavior including the safety re-checks."""
    age_clause_sql: sql.SQL | sql.Composed
    if force_thread:
        age_clause_sql = sql.SQL("TRUE")
    else:
        age_clause_sql = sql.SQL("th.updated_at < now() - make_interval(days => {})").format(
            sql.Literal(int(retention_days))
        )

    common_gates = sql.SQL(
        """
        AND th.status <> ALL(ARRAY['busy']::text[])
        AND ({age_clause})
        AND NOT EXISTS (
            SELECT 1 FROM runs r
            WHERE r.thread_id = th.thread_id
              AND r.status = ANY(ARRAY['pending','running']::text[])
        )
        """
    ).format(age_clause=age_clause_sql)

    counts = {}
    with conn.cursor() as cur:
        log.info("Estimating checkpoint deletion counts...")

        cur.execute(
            sql.SQL(
                """
            SELECT count(*) FROM checkpoint_writes cw
            JOIN threads_to_trim t USING (thread_id, checkpoint_ns)
            JOIN thread th ON th.thread_id = cw.thread_id
            WHERE cw.checkpoint_id <> t.latest_id
            {common_gates}
                """
            ).format(common_gates=common_gates)
        )
        counts["checkpoint_writes"] = int(require_row(cur.fetchone(), context="checkpoint_writes estimate")[0])
        log.info("  checkpoint_writes to delete: %s", f"{counts['checkpoint_writes']:,}")

        cur.execute(
            sql.SQL(
                """
            SELECT count(*) FROM checkpoint_blobs cb
            JOIN threads_to_trim t USING (thread_id, checkpoint_ns)
            JOIN thread th ON th.thread_id = cb.thread_id
            WHERE NOT EXISTS (
                SELECT 1 FROM blobs_to_keep k
                WHERE k.thread_id = cb.thread_id
                  AND k.checkpoint_ns = cb.checkpoint_ns
                  AND k.channel = cb.channel
                  AND k.version = cb.version
            )
            {common_gates}
                """
            ).format(common_gates=common_gates)
        )
        counts["checkpoint_blobs"] = int(require_row(cur.fetchone(), context="checkpoint_blobs estimate")[0])
        log.info("  checkpoint_blobs to delete: %s", f"{counts['checkpoint_blobs']:,}")

        cur.execute(
            sql.SQL(
                """
            SELECT count(*) FROM checkpoints c
            JOIN threads_to_trim t USING (thread_id, checkpoint_ns)
            JOIN thread th ON th.thread_id = c.thread_id
            WHERE c.checkpoint_id <> t.latest_id
            {common_gates}
                """
            ).format(common_gates=common_gates)
        )
        counts["checkpoints"] = int(require_row(cur.fetchone(), context="checkpoints estimate")[0])
        log.info("  checkpoints to delete: %s", f"{counts['checkpoints']:,}")

    return counts


# Batched DELETE statements with per-batch revalidation against thread/runs.

DELETE_WRITES_SQL_TEMPLATE = """
WITH victims AS (
    SELECT cw.thread_id, cw.checkpoint_ns, cw.checkpoint_id, cw.task_id, cw.idx
    FROM checkpoint_writes cw
    JOIN threads_to_trim t USING (thread_id, checkpoint_ns)
    JOIN thread th ON th.thread_id = cw.thread_id
    WHERE cw.checkpoint_id <> t.latest_id
      AND th.status <> ALL(ARRAY['busy']::text[])
      AND ({age_clause})
      AND NOT EXISTS (
          SELECT 1 FROM runs r
          WHERE r.thread_id = cw.thread_id
            AND r.status = ANY(ARRAY['pending','running']::text[])
      )
    LIMIT %s
)
DELETE FROM checkpoint_writes cw
USING victims v
WHERE cw.thread_id     = v.thread_id
  AND cw.checkpoint_ns = v.checkpoint_ns
  AND cw.checkpoint_id = v.checkpoint_id
  AND cw.task_id       = v.task_id
  AND cw.idx           = v.idx
"""

DELETE_BLOBS_SQL_TEMPLATE = """
WITH victims AS (
    SELECT cb.thread_id, cb.checkpoint_ns, cb.channel, cb.version
    FROM checkpoint_blobs cb
    JOIN threads_to_trim t USING (thread_id, checkpoint_ns)
    JOIN thread th ON th.thread_id = cb.thread_id
    WHERE NOT EXISTS (
        SELECT 1 FROM blobs_to_keep k
        WHERE k.thread_id     = cb.thread_id
          AND k.checkpoint_ns = cb.checkpoint_ns
          AND k.channel       = cb.channel
          AND k.version       = cb.version
    )
      AND th.status <> ALL(ARRAY['busy']::text[])
      AND ({age_clause})
      AND NOT EXISTS (
          SELECT 1 FROM runs r
          WHERE r.thread_id = cb.thread_id
            AND r.status = ANY(ARRAY['pending','running']::text[])
      )
    LIMIT %s
)
DELETE FROM checkpoint_blobs cb
USING victims v
WHERE cb.thread_id     = v.thread_id
  AND cb.checkpoint_ns = v.checkpoint_ns
  AND cb.channel       = v.channel
  AND cb.version       = v.version
"""

DELETE_CHECKPOINTS_SQL_TEMPLATE = """
WITH victims AS (
    SELECT c.thread_id, c.checkpoint_ns, c.checkpoint_id
    FROM checkpoints c
    JOIN threads_to_trim t USING (thread_id, checkpoint_ns)
    JOIN thread th ON th.thread_id = c.thread_id
    WHERE c.checkpoint_id <> t.latest_id
      AND th.status <> ALL(ARRAY['busy']::text[])
      AND ({age_clause})
      AND NOT EXISTS (
          SELECT 1 FROM runs r
          WHERE r.thread_id = c.thread_id
            AND r.status = ANY(ARRAY['pending','running']::text[])
      )
    LIMIT %s
)
DELETE FROM checkpoints c
USING victims v
WHERE c.thread_id     = v.thread_id
  AND c.checkpoint_ns = v.checkpoint_ns
  AND c.checkpoint_id = v.checkpoint_id
"""


def build_delete_statements(
    retention_days: int,
    force_thread: bool,
) -> tuple[sql.SQL | sql.Composed, sql.SQL | sql.Composed, sql.SQL | sql.Composed]:
    """Render the three DELETE templates with the proper age clause."""
    age_clause: sql.SQL | sql.Composed
    if force_thread:
        age_clause = sql.SQL("TRUE")
    else:
        age_clause = sql.SQL("th.updated_at < now() - make_interval(days => {})").format(
            sql.Literal(int(retention_days))
        )

    return (
        sql.SQL(DELETE_WRITES_SQL_TEMPLATE).format(age_clause=age_clause),
        sql.SQL(DELETE_BLOBS_SQL_TEMPLATE).format(age_clause=age_clause),
        sql.SQL(DELETE_CHECKPOINTS_SQL_TEMPLATE).format(age_clause=age_clause),
    )


# ---------------------------------------------------------------------------
# Cleanup steps - runs
# ---------------------------------------------------------------------------
def estimate_runs_deletes(
    conn: psycopg.Connection,
    retention_days: int,
    thread_id: str | None = None,
) -> int:
    """Count runs eligible for deletion."""
    where_clauses: list[sql.SQL] = [
        sql.SQL("status = ANY(%s)"),
        sql.SQL("updated_at < now() - make_interval(days => %s)"),
        sql.SQL("(lease_expires_at IS NULL OR lease_expires_at < now())"),
    ]
    params: list = [list(RUN_TERMINAL_STATUSES), retention_days]

    if thread_id:
        where_clauses.append(sql.SQL("thread_id = %s"))
        params.append(thread_id)

    where_sql = sql.SQL(" AND ").join(where_clauses)
    sql_q = sql.SQL("SELECT count(*) FROM runs WHERE {}").format(where_sql)
    with conn.cursor() as cur:
        log.info("Estimating runs deletion count...")
        cur.execute(sql_q, params)
        count = int(require_row(cur.fetchone(), context="runs estimate")[0])
        log.info("  runs to delete: %s", f"{count:,}")
    return count


def delete_runs_in_batches(
    conn: psycopg.Connection,
    retention_days: int,
    batch_size: int,
    pause_seconds: float,
    thread_id: str | None = None,
    max_iterations: int = 100_000,
) -> int:
    """Batched deletion of terminal runs older than retention window."""
    where_clauses: list[sql.SQL] = [
        sql.SQL("status = ANY(%s)"),
        sql.SQL("updated_at < now() - make_interval(days => %s)"),
        sql.SQL("(lease_expires_at IS NULL OR lease_expires_at < now())"),
    ]
    params_template: list = [list(RUN_TERMINAL_STATUSES), retention_days]

    if thread_id:
        where_clauses.append(sql.SQL("thread_id = %s"))
        params_template.append(thread_id)

    where_sql = sql.SQL(" AND ").join(where_clauses)
    delete_sql = sql.SQL(
        """
        WITH victims AS (
            SELECT run_id FROM runs
            WHERE {}
            LIMIT %s
        )
        DELETE FROM runs r USING victims v WHERE r.run_id = v.run_id
        """
    ).format(where_sql)

    log.info("Deleting from runs in batches of %s", f"{batch_size:,}")
    total = 0
    iterations = 0
    started = time.monotonic()

    while iterations < max_iterations:
        iterations += 1
        with conn.cursor() as cur:
            cur.execute(delete_sql, [*params_template, batch_size])
            affected = cur.rowcount
        conn.commit()

        total += affected
        if iterations % 10 == 0 or affected < batch_size:
            elapsed = time.monotonic() - started
            rate = total / elapsed if elapsed > 0 else 0
            log.info("  runs: deleted %s so far (%.0f rows/sec, batch=%s)", f"{total:,}", rate, affected)

        if affected == 0:
            break
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if iterations >= max_iterations:
        log.warning("runs: hit max_iterations safety limit")

    elapsed = time.monotonic() - started
    log.info("  runs: done. Total deleted: %s in %.1fs", f"{total:,}", elapsed)
    return total


# ---------------------------------------------------------------------------
# Generic batched delete
# ---------------------------------------------------------------------------
def delete_in_batches(
    conn: psycopg.Connection,
    label: str,
    delete_sql: sql.SQL | sql.Composed,
    batch_size: int,
    pause_seconds: float,
    max_iterations: int = 100_000,
) -> int:
    log.info("Deleting from %s in batches of %s", label, f"{batch_size:,}")
    total = 0
    iterations = 0
    started = time.monotonic()

    while iterations < max_iterations:
        iterations += 1
        with conn.cursor() as cur:
            cur.execute(delete_sql, (batch_size,))
            affected = cur.rowcount
        conn.commit()

        total += affected
        if iterations % 10 == 0 or affected < batch_size:
            elapsed = time.monotonic() - started
            rate = total / elapsed if elapsed > 0 else 0
            log.info("  %s: deleted %s so far (%.0f rows/sec, batch=%s)", label, f"{total:,}", rate, affected)

        if affected == 0:
            break
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if iterations >= max_iterations:
        log.warning("%s: hit max_iterations safety limit", label)

    elapsed = time.monotonic() - started
    log.info("  %s: done. Total deleted: %s in %.1fs", label, f"{total:,}", elapsed)
    return total


# ---------------------------------------------------------------------------
# Snapshots for thread-scoped testing
# ---------------------------------------------------------------------------
def snapshot_thread(conn: psycopg.Connection, thread_id: str, label: str) -> None:
    """Log a per-table summary for a single thread."""
    with conn.cursor() as cur:
        log.info("--- Thread snapshot: %s (thread_id=%s) ---", label, thread_id)

        cur.execute("SELECT status, updated_at FROM thread WHERE thread_id = %s", (thread_id,))
        row = cur.fetchone()
        if row:
            log.info("  thread:            status=%s updated_at=%s", row[0], row[1])
        else:
            log.info("  thread:            (not found)")

        cur.execute(
            """
            SELECT status, count(*)
            FROM runs WHERE thread_id = %s
            GROUP BY status ORDER BY count(*) DESC
            """,
            (thread_id,),
        )
        rs = cur.fetchall()
        if rs:
            for st, cnt in rs:
                log.info("  runs[%s]: %s", st, f"{cnt:,}")
        else:
            log.info("  runs:              (none)")

        cur.execute(
            """
            SELECT checkpoint_ns, count(*)
            FROM checkpoints WHERE thread_id = %s
            GROUP BY checkpoint_ns ORDER BY checkpoint_ns
            """,
            (thread_id,),
        )
        for ns, cnt in cur.fetchall():
            log.info("  checkpoints[ns=%r]: %s rows", ns, f"{cnt:,}")

        cur.execute(
            """
            SELECT count(*),
                   pg_size_pretty(COALESCE(sum(pg_column_size(blob)),0)::bigint)
            FROM checkpoint_writes WHERE thread_id = %s
            """,
            (thread_id,),
        )
        wcount, wsize = require_row(cur.fetchone(), context="checkpoint_writes snapshot")
        log.info("  checkpoint_writes: %s rows, ~%s on disk", f"{wcount:,}", wsize)

        cur.execute(
            """
            SELECT count(*),
                   pg_size_pretty(COALESCE(sum(pg_column_size(blob)),0)::bigint)
            FROM checkpoint_blobs WHERE thread_id = %s
            """,
            (thread_id,),
        )
        bcount, bsize = require_row(cur.fetchone(), context="checkpoint_blobs snapshot")
        log.info("  checkpoint_blobs:  %s rows, ~%s on disk", f"{bcount:,}", bsize)


def show_kept_checkpoint(conn: psycopg.Connection, thread_id: str) -> None:
    """Show which checkpoint will be retained for a single-thread test."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT checkpoint_ns, MAX(checkpoint_id)
            FROM checkpoints WHERE thread_id = %s
            GROUP BY checkpoint_ns ORDER BY checkpoint_ns
            """,
            (thread_id,),
        )
        for ns, latest_id in cur.fetchall():
            log.info("  KEEP latest checkpoint: ns=%r id=%s", ns, latest_id)
            cur.execute(
                """
                SELECT checkpoint -> 'channel_versions'
                FROM checkpoints
                WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                """,
                (thread_id, ns, latest_id),
            )
            row = cur.fetchone()
            if row and row[0] and isinstance(row[0], dict):
                cv = row[0]
                log.info("    channel_versions has %d entries", len(cv))
                for ch, ver in list(cv.items())[:10]:
                    log.info("      %s -> %s", ch, ver)
                if len(cv) > 10:
                    log.info("      ... and %d more", len(cv) - 10)
            else:
                log.warning(
                    "    NO channel_versions in this checkpoint - "
                    "ALL blobs for this thread will be deleted. "
                    "Verify before running live."
                )


# ---------------------------------------------------------------------------
# Advisory lock
# ---------------------------------------------------------------------------
ADVISORY_LOCK_KEY = 0x6C67636B6E7570


def acquire_lock(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
        return bool(require_row(cur.fetchone(), context="advisory lock")[0])


def release_lock(conn: psycopg.Connection) -> None:
    with suppress(Exception):
        conn.rollback()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
        conn.commit()
    except Exception as e:
        log.warning("Could not release advisory lock cleanly: %s (it will be released on connection close)", e)


# ---------------------------------------------------------------------------
# Pass orchestration
# ---------------------------------------------------------------------------
def run_checkpoint_pass(
    conn: psycopg.Connection,
    args: argparse.Namespace,
    pause_seconds: float,
) -> None:
    log.info("=" * 72)
    log.info("CHECKPOINT CLEANUP PASS")
    log.info("=" * 72)

    if args.thread_id:
        snapshot_thread(conn, args.thread_id, "BEFORE (checkpoints)")

    threads_count, _ = build_checkpoint_working_sets(
        conn,
        retention_days=args.retention_days,
        thread_id=args.thread_id,
        force_thread=args.force_thread,
    )

    if threads_count == 0:
        if args.thread_id and not args.force_thread:
            log.info(
                "Thread %s does not qualify for trimming. Use --force-thread to trim regardless of age.", args.thread_id
            )
        else:
            log.info("No threads need trimming.")
        return

    if args.thread_id:
        show_kept_checkpoint(conn, args.thread_id)

    if args.skip_estimate and not args.dry_run:
        log.info("Skipping deletion estimate (--skip-estimate). Proceeding directly to deletes.")
    else:
        counts = estimate_checkpoint_deletes(
            conn,
            retention_days=args.retention_days,
            force_thread=args.force_thread,
        )
        total = sum(counts.values())
        log.info("Estimated checkpoint deletions: %s rows", f"{total:,}")

        if args.dry_run:
            log.info("DRY RUN: no changes made.")
            return

    delete_writes_sql, delete_blobs_sql, delete_checkpoints_sql = build_delete_statements(
        args.retention_days, args.force_thread
    )

    delete_in_batches(conn, "checkpoint_writes", delete_writes_sql, args.batch_size, pause_seconds)
    delete_in_batches(conn, "checkpoint_blobs", delete_blobs_sql, args.batch_size, pause_seconds)
    delete_in_batches(conn, "checkpoints", delete_checkpoints_sql, args.batch_size, pause_seconds)

    if args.thread_id:
        snapshot_thread(conn, args.thread_id, "AFTER (checkpoints)")


def run_runs_pass(
    conn: psycopg.Connection,
    args: argparse.Namespace,
    pause_seconds: float,
) -> None:
    log.info("=" * 72)
    log.info("RUNS CLEANUP PASS")
    log.info("=" * 72)

    if args.skip_estimate and not args.dry_run:
        log.info("Skipping runs deletion estimate (--skip-estimate).")
    else:
        count = estimate_runs_deletes(
            conn,
            retention_days=args.runs_retention_days,
            thread_id=args.thread_id,
        )

        if count == 0:
            log.info("No runs need cleanup.")
            return

        if args.dry_run:
            log.info("DRY RUN: no changes made.")
            return

    delete_runs_in_batches(
        conn,
        retention_days=args.runs_retention_days,
        batch_size=args.batch_size,
        pause_seconds=pause_seconds,
        thread_id=args.thread_id,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only, no deletes")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=7,
        help="Checkpoint retention: trim threads inactive longer than this (default: 7)",
    )
    parser.add_argument(
        "--runs-retention-days",
        type=int,
        default=30,
        help="Runs retention: delete terminal runs older than this (default: 30)",
    )
    parser.add_argument("--batch-size", type=int, default=10_000, help="Rows per DELETE batch (default: 10000)")
    parser.add_argument("--pause-ms", type=int, default=100, help="Pause between batches in ms (default: 100)")
    parser.add_argument(
        "--statement-timeout-ms", type=int, default=300_000, help="Per-statement timeout in ms (default: 300000)"
    )
    parser.add_argument("--thread-id", type=str, default=None, help="Scope cleanup to a single thread_id (for testing)")
    parser.add_argument(
        "--force-thread", action="store_true", help="With --thread-id, ignore age cutoff. Testing only."
    )
    parser.add_argument("--skip-checkpoints", action="store_true", help="Skip the checkpoint cleanup pass")
    parser.add_argument("--skip-runs", action="store_true", help="Skip the runs cleanup pass")
    parser.add_argument(
        "--skip-estimate",
        action="store_true",
        help="Skip the deletion-count estimate before running. "
        "Useful on large databases where the count query is "
        "slow. The actual deletes still run; you just don't "
        "get a preview of how many rows will be affected. "
        "Ignored when --dry-run is set (dry-run needs the "
        "estimate to do anything useful).",
    )
    args = parser.parse_args()

    if args.force_thread and not args.thread_id:
        log.error("--force-thread requires --thread-id")
        return 2
    if args.skip_checkpoints and args.skip_runs:
        log.error("--skip-checkpoints and --skip-runs together = nothing to do")
        return 2

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log.error("DATABASE_URL environment variable not set")
        return 2

    log.info("=" * 72)
    log.info("LangGraph cleanup")
    log.info("  checkpoint retention: %s days", args.retention_days)
    log.info("  runs retention:       %s days", args.runs_retention_days)
    log.info("  batch:                %s rows, %sms pause", f"{args.batch_size:,}", args.pause_ms)
    log.info("  mode:                 %s", "DRY RUN" if args.dry_run else "EXECUTE")
    if args.thread_id:
        log.info("  scope:                thread_id=%s%s", args.thread_id, " (force)" if args.force_thread else "")
    if args.skip_checkpoints:
        log.info("  skipping checkpoint pass")
    if args.skip_runs:
        log.info("  skipping runs pass")
    if args.skip_estimate:
        log.info("  skipping deletion estimate (--skip-estimate)")
    log.info("=" * 72)

    started = time.monotonic()
    pause_seconds = args.pause_ms / 1000.0

    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET statement_timeout = {}").format(sql.Literal(args.statement_timeout_ms)))
            cur.execute("SET lock_timeout = '5s'")
            cur.execute("SET idle_in_transaction_session_timeout = '60s'")
        conn.commit()

        if not acquire_lock(conn):
            log.error("Another cleanup job is already running. Exiting.")
            return 3

        try:
            if not args.skip_checkpoints:
                run_checkpoint_pass(conn, args, pause_seconds)
            if not args.skip_runs:
                run_runs_pass(conn, args, pause_seconds)
        finally:
            release_lock(conn)

    elapsed = time.monotonic() - started
    log.info("Cleanup complete in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
