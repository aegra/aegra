"""add_is_ephemeral_to_thread

Adds a boolean marker so the orphan-thread sweeper can find ephemeral
threads (created by the stateless run endpoints) whose fast-path
delete-after-run was missed, without touching threads created via
POST /threads. Column add uses a fixed default, which Postgres 11+
applies as metadata-only (no table rewrite). The partial index is built
CONCURRENTLY, outside the migration's transaction, to avoid a SHARE lock
on `thread` for the duration of the build (same rationale as
d9e0f1a23456).

Revision ID: 892140cab119
Revises: a3f7c1d9e2b4
Create Date: 2026-09-15 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "892140cab119"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None

INDEX_NAME = "idx_thread_ephemeral_updated_at"


def upgrade() -> None:
    # IF NOT EXISTS: autocommit_block below commits this column add before
    # the index build, ahead of the revision record — a retry after a failed
    # or interrupted build must not fail on "column already exists".
    op.execute("ALTER TABLE thread ADD COLUMN IF NOT EXISTS is_ephemeral BOOLEAN NOT NULL DEFAULT false")
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(f"CREATE INDEX CONCURRENTLY {INDEX_NAME} ON thread (updated_at) WHERE is_ephemeral")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
    op.execute("ALTER TABLE thread DROP COLUMN IF EXISTS is_ephemeral")
