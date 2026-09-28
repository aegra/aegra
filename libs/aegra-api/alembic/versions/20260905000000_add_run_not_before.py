"""add persisted delayed run scheduling

Revision ID: 6b2d9f1a4c7e
Revises: a3f7c1d9e2b4
Create Date: 2026-09-05 00:00:00.000000
"""

import sqlalchemy as sa

from alembic import op

revision = "6b2d9f1a4c7e"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("not_before", sa.TIMESTAMP(timezone=True), nullable=True))
    op.create_index("idx_runs_not_before", "runs", ["status", "not_before"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_runs_not_before", table_name="runs")
    op.drop_column("runs", "not_before")
