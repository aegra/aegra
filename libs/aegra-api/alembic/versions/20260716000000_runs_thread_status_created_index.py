"""runs: composite index for multitask per-thread admission and dispatch

Revision ID: d2e3f4a5b6c7
Revises: b88bb61be638
Create Date: 2026-07-16 00:00:00.000000

Adds a composite index on runs(thread_id, status, created_at) so the
double-texting admission check (does this thread have an active/queued run?)
and the FIFO queued-run dispatch (oldest queued run for a thread) are
index-only instead of scanning all of a thread's runs.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "d2e3f4a5b6c7"
down_revision = "b88bb61be638"
branch_labels = None
depends_on = None


INDEX_NAME = "idx_runs_thread_status_created"


def upgrade() -> None:
    # CONCURRENTLY so the build holds only SHARE UPDATE EXCLUSIVE (not SHARE) on
    # the hot runs table — migrations run at lifespan startup on every deploy.
    # DROP first: an interrupted CONCURRENTLY build leaves an INVALID index that
    # IF NOT EXISTS would skip forever, so the retry must rebuild from scratch.
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(f"CREATE INDEX CONCURRENTLY {INDEX_NAME} ON runs (thread_id, status, created_at)")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
