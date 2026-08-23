"""Add claim_token to runs for per-acquisition lease fencing

``claimed_by`` is a stable worker name (hostname-pid-worker-idx), so it is
reused across acquisitions. A worker that stalls past its lease, gets reaped,
and then re-acquires the same run still matches its own ownership predicates
and can extend, release, or terminalize the replacement attempt (#502).

``claim_token`` is regenerated on every acquisition, giving each attempt an
identity that a stale attempt cannot forge.

Revision ID: c9d0e1f2a345
Revises: b88bb61be638
Create Date: 2026-08-23 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "c9d0e1f2a345"
down_revision = "b88bb61be638"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("claim_token", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "claim_token")
