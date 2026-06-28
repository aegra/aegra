"""Replace verbatim-config btree index with md5 expression index.

The unique btree index ``idx_assistant_user_graph_config`` was introduced in
revision ``76dfdbe90d2b`` (add_version_table) as::

    CREATE UNIQUE INDEX idx_assistant_user_graph_config
        ON assistant (user_id, graph_id, config);

PostgreSQL btree index entries are capped at roughly **2 704 bytes** (one
third of an 8 KB page). The ``config`` column is a JSONB object whose
``configurable`` sub-key holds agent configuration — including a system
prompt that routinely exceeds 2 704 bytes. Every ``INSERT`` or ``UPDATE``
whose serialised ``config`` is larger than that limit fails with::

    index row size NNNN exceeds btree version 4 maximum 2704 for index
    "idx_assistant_user_graph_config"

This migration replaces the verbatim-column btree with a fixed-width
**expression index** over ``md5(config::text)``::

    CREATE UNIQUE INDEX idx_assistant_user_graph_config
        ON assistant (user_id, graph_id, md5(config::text));

``md5(config::text)`` is always 32 bytes, so the index entry never exceeds
the btree limit regardless of payload size.  The uniqueness guarantee is
preserved: PostgreSQL stores JSONB in canonical form (normalised key order,
no insignificant whitespace), so ``config::text`` is deterministic for
logically-equal configs.  ``md5`` is a PostgreSQL built-in — no
``pgcrypto`` extension is required.

Downgrade note
--------------
The downgrade re-creates the original btree index.  This will fail on any
database that already contains a row whose ``config::text`` exceeds
~2 704 bytes (i.e. the exact problem this migration fixes).  Treat this
migration as **forward-only** on production databases that have been used
with large agent configs.

Revision ID: b88bb61be638
Revises: c7d1f2a4b6e8
Create Date: 2026-06-12 00:00:00.000000
"""

from alembic import op

revision = "b88bb61be638"
down_revision = "c7d1f2a4b6e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("idx_assistant_user_graph_config", table_name="assistant")
    # CREATE INDEX cannot use CONCURRENTLY inside a transaction, and Alembic
    # migrations run inside a transaction by default.  The brief ShareLock on
    # the assistant table during the index build is acceptable: the table is
    # small and this migration runs once at deployment time.
    op.execute(
        "CREATE UNIQUE INDEX idx_assistant_user_graph_config ON assistant (user_id, graph_id, md5(config::text))"
    )


def downgrade() -> None:
    # WARNING: re-creating the verbatim-config btree will fail if any row's
    # config::text already exceeds ~2 704 bytes.  Only safe on databases that
    # have never stored a large-config assistant.
    op.drop_index("idx_assistant_user_graph_config", table_name="assistant")
    op.create_index(
        "idx_assistant_user_graph_config",
        "assistant",
        ["user_id", "graph_id", "config"],
        unique=True,
    )
