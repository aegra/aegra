"""cron: cascade thread FK, add claimed_until column

Revision ID: c7d1f2a4b6e8
Revises: baa02a5d050f
Create Date: 2026-05-04 00:00:00.000000

Two changes to the ``crons`` table:

* ``thread_id`` foreign key changes from ``ON DELETE SET NULL`` to
  ``ON DELETE CASCADE``. A cron bound to a thread that gets deleted is
  removed atomically instead of silently flipping to a stateless cron.
* New ``claimed_until`` column tracks an explicit scheduler lease so the
  claim window is decoupled from the poll interval. When ``NULL`` the cron
  is not currently claimed; when set in the future, the cron is treated as
  in-flight and skipped by other pollers.
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "c7d1f2a4b6e8"
down_revision = "baa02a5d050f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Replace SET NULL FK with CASCADE.
    op.execute("ALTER TABLE crons DROP CONSTRAINT IF EXISTS crons_thread_id_fkey")
    op.create_foreign_key(
        "crons_thread_id_fkey",
        "crons",
        "thread",
        ["thread_id"],
        ["thread_id"],
        ondelete="CASCADE",
    )

    # New claimed_until lease column.
    op.add_column(
        "crons",
        sa.Column("claimed_until", sa.TIMESTAMP(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("crons", "claimed_until")

    op.execute("ALTER TABLE crons DROP CONSTRAINT IF EXISTS crons_thread_id_fkey")
    op.create_foreign_key(
        "crons_thread_id_fkey",
        "crons",
        "thread",
        ["thread_id"],
        ["thread_id"],
        ondelete="SET NULL",
    )
