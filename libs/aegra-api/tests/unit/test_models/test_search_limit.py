"""Validation tests for threads/store search `limit` (issue #471)."""

from typing import Any

import pytest
from pydantic import ValidationError

from aegra_api.models.search_limit import (
    DEFAULT_SEARCH_LIMIT,
    effective_search_limit,
    resolve_search_limit,
    search_limit_json_schema_extra,
)
from aegra_api.models.store import StoreSearchRequest
from aegra_api.models.threads import ThreadSearchRequest
from aegra_api.settings import settings

_DEFAULT_CAP: int = 1000
_INVALID_LIMITS: tuple[Any, ...] = ("abc", [], {}, 20.5)
_SearchModel = type[ThreadSearchRequest] | type[StoreSearchRequest]


@pytest.fixture(autouse=True)
def _pin_default_search_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", _DEFAULT_CAP)


def _integer_limit_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the integer branch of an optional limit field schema."""
    for option in schema.get("anyOf", []):
        if isinstance(option, dict) and option.get("type") == "integer":
            return option
    return schema


def _limit_error(exc: ValidationError) -> dict[str, Any]:
    errors = [item for item in exc.errors() if item["loc"][-1] == "limit"]
    assert len(errors) == 1
    return errors[0]


def _store_request(**kwargs: Any) -> StoreSearchRequest:
    return StoreSearchRequest(namespace_prefix=["notes"], **kwargs)


def _build(
    model: _SearchModel,
    *,
    data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ThreadSearchRequest | StoreSearchRequest:
    payload: dict[str, Any] = dict(data if data is not None else kwargs)
    if model is StoreSearchRequest:
        payload.setdefault("namespace_prefix", ["notes"])
    if data is not None:
        return model.model_validate(payload)
    return model(**payload)


class TestResolveSearchLimit:
    """Shared helper used by both search request models."""

    def test_none_resolves_to_default_when_cap_is_higher(self) -> None:
        assert resolve_search_limit(None) == DEFAULT_SEARCH_LIMIT
        assert resolve_search_limit(None) == 20

    def test_none_resolves_to_cap_when_cap_is_below_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        assert resolve_search_limit(None) == 10
        assert effective_search_limit() == 10

    def test_passes_value_at_cap(self) -> None:
        assert resolve_search_limit(_DEFAULT_CAP) == _DEFAULT_CAP

    def test_schema_extra_sets_maximum_on_integer_anyof_branch(self) -> None:
        schema: dict[str, Any] = {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}
        search_limit_json_schema_extra(schema)
        assert schema["anyOf"][0]["maximum"] == _DEFAULT_CAP
        assert schema["default"] == DEFAULT_SEARCH_LIMIT
        assert "maximum" not in schema

    def test_schema_extra_sets_maximum_without_anyof(self) -> None:
        schema: dict[str, Any] = {"type": "integer"}
        search_limit_json_schema_extra(schema)
        assert schema["maximum"] == _DEFAULT_CAP
        assert schema["default"] == DEFAULT_SEARCH_LIMIT

    def test_schema_extra_tracks_live_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        schema: dict[str, Any] = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        search_limit_json_schema_extra(schema)
        assert schema["anyOf"][0]["maximum"] == 50
        assert schema["default"] == DEFAULT_SEARCH_LIMIT


class TestThreadSearchRequestLimit:
    """ThreadSearchRequest.limit must accept LangGraph SDK page sizes."""

    @pytest.mark.parametrize("limit", [1, 20, 101, 500, 1000])
    def test_accepts_limits_within_default_cap(self, limit: int) -> None:
        request = ThreadSearchRequest(limit=limit)
        assert request.limit == limit

    def test_accepts_limit_equal_to_cap(self) -> None:
        request = ThreadSearchRequest(limit=settings.app.MAX_SEARCH_LIMIT)
        assert request.limit == settings.app.MAX_SEARCH_LIMIT

    def test_omitted_limit_uses_default(self) -> None:
        request = ThreadSearchRequest()
        assert request.limit == DEFAULT_SEARCH_LIMIT

    def test_omitted_limit_via_model_validate_uses_default(self) -> None:
        request = ThreadSearchRequest.model_validate({})
        assert request.limit == DEFAULT_SEARCH_LIMIT

    def test_explicit_none_limit_uses_default(self) -> None:
        request = ThreadSearchRequest(limit=None)
        assert request.limit == DEFAULT_SEARCH_LIMIT

    def test_explicit_none_limit_via_model_validate_uses_default(self) -> None:
        request = ThreadSearchRequest.model_validate({"limit": None})
        assert request.limit == DEFAULT_SEARCH_LIMIT

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_zero_and_negative_limits(self, limit: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest(limit=limit)
        err = _limit_error(exc_info.value)
        assert err["type"] == "greater_than_equal"

    def test_rejects_limit_above_cap(self) -> None:
        cap = settings.app.MAX_SEARCH_LIMIT
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest(limit=cap + 1)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": cap}
        assert str(cap) in err["msg"]

    @pytest.mark.parametrize("limit", _INVALID_LIMITS)
    def test_rejects_non_integer_limits(self, limit: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest.model_validate({"limit": limit})
        err = _limit_error(exc_info.value)
        assert err["type"] in {"int_parsing", "int_type", "int_from_float"}

    def test_default_limit_is_unchanged(self) -> None:
        request = ThreadSearchRequest()
        assert request.limit == DEFAULT_SEARCH_LIMIT
        assert DEFAULT_SEARCH_LIMIT == 20
        assert DEFAULT_SEARCH_LIMIT != settings.app.MAX_SEARCH_LIMIT

    def test_honors_configured_max_search_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        request = ThreadSearchRequest(limit=50)
        assert request.limit == 50
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest(limit=51)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": 50}

    def test_openapi_schema_advertises_configured_cap(self) -> None:
        schema = ThreadSearchRequest.model_json_schema()["properties"]["limit"]
        integer_schema = _integer_limit_schema(schema)
        assert integer_schema["minimum"] == 1
        assert integer_schema["maximum"] == settings.app.MAX_SEARCH_LIMIT
        assert schema["default"] == DEFAULT_SEARCH_LIMIT

    def test_openapi_schema_tracks_live_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        schema = ThreadSearchRequest.model_json_schema()["properties"]["limit"]
        assert _integer_limit_schema(schema)["maximum"] == 50
        assert schema["default"] == DEFAULT_SEARCH_LIMIT

    def test_omitted_and_null_honor_cap_below_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        assert ThreadSearchRequest().limit == 10
        assert ThreadSearchRequest.model_validate({}).limit == 10
        assert ThreadSearchRequest(limit=None).limit == 10
        assert ThreadSearchRequest.model_validate({"limit": None}).limit == 10
        assert ThreadSearchRequest(limit=10).limit == 10
        with pytest.raises(ValidationError) as exc_info:
            ThreadSearchRequest(limit=11)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": 10}

    def test_openapi_schema_default_is_capped_when_cap_below_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        schema = ThreadSearchRequest.model_json_schema()["properties"]["limit"]
        assert schema["default"] == 10
        assert _integer_limit_schema(schema)["maximum"] == 10


class TestStoreSearchRequestLimit:
    """StoreSearchRequest.limit shares the same cap as thread search."""

    @pytest.mark.parametrize("limit", [1, 20, 101, 500, 1000])
    def test_accepts_limits_within_default_cap(self, limit: int) -> None:
        request = _store_request(limit=limit)
        assert request.limit == limit

    def test_accepts_limit_equal_to_cap(self) -> None:
        request = _store_request(limit=settings.app.MAX_SEARCH_LIMIT)
        assert request.limit == settings.app.MAX_SEARCH_LIMIT

    def test_omitted_limit_uses_default(self) -> None:
        request = _store_request()
        assert request.limit == DEFAULT_SEARCH_LIMIT

    def test_omitted_limit_via_model_validate_uses_default(self) -> None:
        request = StoreSearchRequest.model_validate({"namespace_prefix": ["notes"]})
        assert request.limit == DEFAULT_SEARCH_LIMIT

    def test_explicit_none_limit_uses_default(self) -> None:
        request = _store_request(limit=None)
        assert request.limit == DEFAULT_SEARCH_LIMIT

    def test_explicit_none_limit_via_model_validate_uses_default(self) -> None:
        request = StoreSearchRequest.model_validate({"namespace_prefix": ["notes"], "limit": None})
        assert request.limit == DEFAULT_SEARCH_LIMIT

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_zero_and_negative_limits(self, limit: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _store_request(limit=limit)
        err = _limit_error(exc_info.value)
        assert err["type"] == "greater_than_equal"

    def test_rejects_limit_above_cap(self) -> None:
        cap = settings.app.MAX_SEARCH_LIMIT
        with pytest.raises(ValidationError) as exc_info:
            _store_request(limit=cap + 1)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": cap}
        assert str(cap) in err["msg"]

    @pytest.mark.parametrize("limit", _INVALID_LIMITS)
    def test_rejects_non_integer_limits(self, limit: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            StoreSearchRequest.model_validate({"namespace_prefix": ["notes"], "limit": limit})
        err = _limit_error(exc_info.value)
        assert err["type"] in {"int_parsing", "int_type", "int_from_float"}

    def test_default_limit_is_unchanged(self) -> None:
        request = _store_request()
        assert request.limit == DEFAULT_SEARCH_LIMIT
        assert DEFAULT_SEARCH_LIMIT == 20
        assert DEFAULT_SEARCH_LIMIT != settings.app.MAX_SEARCH_LIMIT

    def test_honors_configured_max_search_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        request = _store_request(limit=50)
        assert request.limit == 50
        with pytest.raises(ValidationError) as exc_info:
            _store_request(limit=51)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": 50}

    def test_openapi_schema_advertises_configured_cap(self) -> None:
        schema = StoreSearchRequest.model_json_schema()["properties"]["limit"]
        integer_schema = _integer_limit_schema(schema)
        assert integer_schema["minimum"] == 1
        assert integer_schema["maximum"] == settings.app.MAX_SEARCH_LIMIT
        assert schema["default"] == DEFAULT_SEARCH_LIMIT

    def test_openapi_schema_tracks_live_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 50)
        schema = StoreSearchRequest.model_json_schema()["properties"]["limit"]
        assert _integer_limit_schema(schema)["maximum"] == 50
        assert schema["default"] == DEFAULT_SEARCH_LIMIT

    def test_omitted_and_null_honor_cap_below_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        assert _store_request().limit == 10
        assert StoreSearchRequest.model_validate({"namespace_prefix": ["notes"]}).limit == 10
        assert _store_request(limit=None).limit == 10
        assert StoreSearchRequest.model_validate({"namespace_prefix": ["notes"], "limit": None}).limit == 10
        assert _store_request(limit=10).limit == 10
        with pytest.raises(ValidationError) as exc_info:
            _store_request(limit=11)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": 10}

    def test_openapi_schema_default_is_capped_when_cap_below_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        schema = StoreSearchRequest.model_json_schema()["properties"]["limit"]
        assert schema["default"] == 10
        assert _integer_limit_schema(schema)["maximum"] == 10


@pytest.mark.parametrize("model", [ThreadSearchRequest, StoreSearchRequest], ids=["threads", "store"])
class TestSearchLimitParity:
    """Thread and store search requests share one resolution table."""

    def test_default_cap_omitted_and_null_are_twenty(self, model: _SearchModel) -> None:
        omitted = _build(model, data={})
        constructed = _build(model)
        explicit_null = _build(model, data={"limit": None})
        assert omitted.limit == constructed.limit == explicit_null.limit == 20

    def test_cap_ten_omitted_null_and_ten_are_ten(self, model: _SearchModel, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        omitted = _build(model, data={})
        constructed = _build(model)
        explicit_null = _build(model, limit=None)
        at_cap = _build(model, limit=10)
        assert omitted.limit == constructed.limit == explicit_null.limit == at_cap.limit == 10

    def test_cap_ten_rejects_eleven(self, model: _SearchModel, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        with pytest.raises(ValidationError) as exc_info:
            _build(model, limit=11)
        err = _limit_error(exc_info.value)
        assert err["type"] == "less_than_equal"
        assert err["ctx"] == {"le": 10}

    def test_openapi_default_and_maximum_match_runtime(
        self, model: _SearchModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.app, "MAX_SEARCH_LIMIT", 10)
        schema = model.model_json_schema()["properties"]["limit"]
        assert schema["default"] == effective_search_limit()
        assert _integer_limit_schema(schema)["maximum"] == settings.app.MAX_SEARCH_LIMIT
