"""replace uuid-ossp with gen_random_uuid

Requires PostgreSQL 13 or later.

uuid_generate_v4() requires the uuid-ossp extension; gen_random_uuid() is
built into PostgreSQL core (since v13) and produces identical v4 UUIDs.
Removing the extension dependency simplifies deployment on managed Postgres
services that restrict extension installation.

Revision ID: 4c77fafdc3b0
Revises: e0f1a234b567
Create Date: 2026-05-13 09:55:25.178674

"""

from alembic import op

revision = "4c77fafdc3b0"
down_revision = "e0f1a234b567"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE assistant ALTER COLUMN assistant_id SET DEFAULT gen_random_uuid()::text")
    op.execute("ALTER TABLE runs ALTER COLUMN run_id SET DEFAULT gen_random_uuid()::text")


def downgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("ALTER TABLE assistant ALTER COLUMN assistant_id SET DEFAULT public.uuid_generate_v4()::text")
    op.execute("ALTER TABLE runs ALTER COLUMN run_id SET DEFAULT public.uuid_generate_v4()::text")
