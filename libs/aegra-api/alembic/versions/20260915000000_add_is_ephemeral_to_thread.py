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

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "892140cab119"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None

INDEX_NAME = "idx_thread_ephemeral_updated_at"


def upgrade() -> None:
    op.add_column(
        "thread",
        sa.Column("is_ephemeral", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(f"CREATE INDEX CONCURRENTLY {INDEX_NAME} ON thread (updated_at) WHERE is_ephemeral")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
    op.drop_column("thread", "is_ephemeral")
