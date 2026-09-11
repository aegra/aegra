"""add_langsmith_session_name

Revision ID: c4f2a97b8d10
Revises: a3f7c1d9e2b4
Create Date: 2026-09-01 00:00:00.000000
"""

import sqlalchemy as sa

from alembic import op

revision = "c4f2a97b8d10"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("langsmith_session_name", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "langsmith_session_name")
