"""Unit tests for _resolve_sort in /threads/search."""

from aegra_api.api.threads import _resolve_sort
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import ThreadSearchRequest


def _col_name(column: object) -> str:
    """Return the ORM column's logical name for assertion comparisons."""
    return getattr(column, "key", None) or getattr(column, "name", "")


class TestResolveSortOrderBy:
    """_resolve_sort parses the legacy order_by single-string form."""

    def test_defaults_to_created_at_desc_when_empty(self) -> None:
        """No order_by → default created_at DESC."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by=None))
        assert _col_name(column) == "created_at"
        assert asc is False

    def test_parses_order_by_asc(self) -> None:
        """'<col> ASC' is parsed as ascending."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by="updated_at ASC"))
        assert _col_name(column) == "updated_at"
        assert asc is True

    def test_parses_order_by_desc(self) -> None:
        """'<col> DESC' is parsed as descending."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by="thread_id DESC"))
        assert _col_name(column) == "thread_id"
        assert asc is False

    def test_column_only_defaults_to_desc(self) -> None:
        """A bare column name defaults to descending."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by="status"))
        assert _col_name(column) == "status"
        assert asc is False

    def test_case_insensitive(self) -> None:
        """Column and direction tokens are matched case-insensitively."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by="UPDATED_AT asc"))
        assert _col_name(column) == "updated_at"
        assert asc is True

    def test_falls_back_on_unknown_column(self) -> None:
        """Unknown column name falls back to the default ordering."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by="nonexistent_col"))
        assert _col_name(column) == "created_at"
        assert asc is False

    def test_falls_back_on_sql_injection_attempt(self) -> None:
        """SQL-injection-style input falls back to default — no getattr leak."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by="password; DROP TABLE users --"))
        assert _col_name(column) == "created_at"
        assert asc is False

    def test_falls_back_on_empty_string(self) -> None:
        """Empty string falls back to default (not a crash, not a 500)."""
        column, asc = _resolve_sort(ThreadSearchRequest(order_by=""))
        assert _col_name(column) == "created_at"
        assert asc is False

    def test_returns_real_orm_column(self) -> None:
        """The returned column is the actual ORM attribute, not a string proxy."""
        column, _ = _resolve_sort(ThreadSearchRequest(order_by="updated_at ASC"))
        assert column is ThreadORM.updated_at


class TestResolveSortSdkShape:
    """_resolve_sort honours the SDK-style sort_by / sort_order fields."""

    def test_sdk_shape_asc(self) -> None:
        """SDK sort_by + sort_order='asc' produces ascending ordering."""
        column, asc = _resolve_sort(ThreadSearchRequest(sort_by="updated_at", sort_order="asc"))
        assert _col_name(column) == "updated_at"
        assert asc is True

    def test_sdk_shape_desc(self) -> None:
        """SDK sort_by + sort_order='desc' produces descending ordering."""
        column, asc = _resolve_sort(ThreadSearchRequest(sort_by="thread_id", sort_order="desc"))
        assert _col_name(column) == "thread_id"
        assert asc is False

    def test_sdk_sort_by_defaults_to_desc(self) -> None:
        """sort_by without sort_order defaults to descending."""
        column, asc = _resolve_sort(ThreadSearchRequest(sort_by="updated_at"))
        assert _col_name(column) == "updated_at"
        assert asc is False

    def test_sort_by_takes_precedence_over_order_by(self) -> None:
        """When both sort_by and order_by are set, sort_by wins."""
        column, asc = _resolve_sort(ThreadSearchRequest(sort_by="updated_at", order_by="thread_id ASC"))
        assert _col_name(column) == "updated_at"
        assert asc is False

    def test_sdk_state_updated_at_falls_back_silently(self) -> None:
        """state_updated_at is a valid SDK literal but has no matching column → default."""
        column, asc = _resolve_sort(ThreadSearchRequest(sort_by="state_updated_at", sort_order="asc"))
        assert _col_name(column) == "created_at"
        assert asc is False

    def test_sdk_unknown_sort_by_falls_back(self) -> None:
        """Malformed sort_by falls back silently — no 500, no getattr leak."""
        column, asc = _resolve_sort(ThreadSearchRequest(sort_by="password; DROP TABLE", sort_order="asc"))
        assert _col_name(column) == "created_at"
        assert asc is False
