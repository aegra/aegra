"""Unit tests for the JSONB merge SQL helpers.

The merge exists so that Postgres, not Python, reads the column being written:
these tests pin the rendered statement and the bind type. What the statement
then does under concurrency is covered by
``tests/integration/test_thread_metadata_merge_db.py``, which needs a database.
"""

from sqlalchemy import update
from sqlalchemy.dialects import postgresql

from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.utils.jsonb import jsonb_patch, jsonb_shallow_merge


def _compile(expression) -> str:
    return str(
        update(ThreadORM)
        .where(ThreadORM.thread_id == "t")
        .values(metadata_json=expression)
        .compile(dialect=postgresql.dialect())
    )


class TestJsonbPatch:
    def test_binds_under_the_given_name(self) -> None:
        param = jsonb_patch({"a": 1}, "metadata_patch")
        assert param.key == "metadata_patch"
        assert param.value == {"a": 1}

    def test_bind_strips_null_bytes(self) -> None:
        """Postgres rejects U+0000 in jsonb; the patch binds through JsonbSafe."""
        param = jsonb_patch({"note": "be\x00fore"}, "metadata_patch")
        assert param.type.process_bind_param(param.value, postgresql.dialect()) == {"note": "before"}


class TestJsonbShallowMerge:
    def test_merges_in_sql_rather_than_in_python(self) -> None:
        sql = _compile(jsonb_shallow_merge(ThreadORM.metadata_json, jsonb_patch({"a": 1}, "metadata_patch")))
        assert "THEN thread.metadata_json ELSE '{}'::jsonb END || %(metadata_patch)s::JSONB" in sql

    def test_non_object_metadata_starts_from_an_empty_object(self) -> None:
        """``'[1,2]'::jsonb || '{"a":1}'::jsonb`` appends instead of merging."""
        sql = _compile(jsonb_shallow_merge(ThreadORM.metadata_json, jsonb_patch({"a": 1}, "metadata_patch")))
        assert "CASE WHEN (jsonb_typeof(thread.metadata_json) = %(jsonb_typeof_1)s)" in sql
        assert "ELSE '{}'::jsonb END" in sql

    def test_later_patches_are_applied_last(self) -> None:
        sql = _compile(
            jsonb_shallow_merge(
                ThreadORM.metadata_json,
                jsonb_patch({"a": 1}, "first_patch"),
                jsonb_patch({"a": 2}, "second_patch"),
            )
        )
        assert sql.index("%(first_patch)s") < sql.index("%(second_patch)s")
