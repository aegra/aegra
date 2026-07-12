"""add encrypted secrets column to assistant

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-07-12

Per-assistant secrets are stored Fernet-encrypted in assistant.secrets ({name:
token}) and never returned in responses. Replaces storing raw api_key in
config/context. Nullable-free with a '{}' default so existing rows migrate cleanly.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: str | Sequence[str] | None = "d1e2f3a4b5c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "assistant",
        sa.Column("secrets", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("assistant", "secrets")
