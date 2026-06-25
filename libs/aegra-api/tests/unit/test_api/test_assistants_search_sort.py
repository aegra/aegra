"""Unit tests for _resolve_sort in /assistants/search.

Auth-handler filter merging moved out of the route layer into
build_metadata_filter (see tests/unit/test_core/test_auth_filters.py) and
AssistantService (see test_assistant_service.py::TestAuthDispatch).
"""

import pytest
from pydantic import ValidationError

from aegra_api.api.assistants import _resolve_sort
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.models import AssistantSearchRequest


def _col_name(column: object) -> str:
    return getattr(column, "key", None) or getattr(column, "name", "")


class TestResolveSort:
    """_resolve_sort honours sort_by / sort_order."""

    def test_default_is_created_at_desc(self) -> None:
        column, asc = _resolve_sort(AssistantSearchRequest())
        assert _col_name(column) == "created_at"
        assert asc is False

    def test_sort_by_defaults_to_desc(self) -> None:
        column, asc = _resolve_sort(AssistantSearchRequest(sort_by="updated_at"))
        assert _col_name(column) == "updated_at"
        assert asc is False

    def test_sort_by_asc(self) -> None:
        column, asc = _resolve_sort(AssistantSearchRequest(sort_by="name", sort_order="asc"))
        assert _col_name(column) == "name"
        assert asc is True

    def test_sort_by_desc_explicit(self) -> None:
        column, asc = _resolve_sort(AssistantSearchRequest(sort_by="assistant_id", sort_order="desc"))
        assert _col_name(column) == "assistant_id"
        assert asc is False

    def test_returns_real_orm_column(self) -> None:
        column, _ = _resolve_sort(AssistantSearchRequest(sort_by="updated_at"))
        assert column is AssistantORM.updated_at


class TestSortByValidation:
    """Pydantic validates sort_by against the Literal at request boundary."""

    def test_invalid_sort_by_raises(self) -> None:
        with pytest.raises(ValidationError):
            AssistantSearchRequest(sort_by="password; DROP TABLE assistants --")  # type: ignore[arg-type]

    def test_invalid_sort_order_raises(self) -> None:
        with pytest.raises(ValidationError):
            AssistantSearchRequest(sort_by="name", sort_order="sideways")  # type: ignore[arg-type]
