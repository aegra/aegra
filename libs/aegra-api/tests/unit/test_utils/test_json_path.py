"""Unit tests for thread search extract JSON-path helpers."""

import pytest
from pydantic import ValidationError

from aegra_api.models.threads import ThreadSearchRequest
from aegra_api.utils.json_path import (
    resolve_json_path,
    sources_needed_by_extract,
    validate_extract_path,
    validate_extract_paths,
)


def test_validate_extract_path_accepts_supported_forms() -> None:
    validate_extract_path("values.messages[0].content")
    validate_extract_path("values.messages[-1].content")
    validate_extract_path("metadata.title")
    validate_extract_path("config.configurable.thread_id")


def test_validate_extract_path_rejects_interrupts_root() -> None:
    with pytest.raises(ValueError, match="must start with"):
        validate_extract_path("interrupts.items[0].value")


def test_validate_extract_path_rejects_bad_prefix() -> None:
    with pytest.raises(ValueError, match="must start with"):
        validate_extract_path("state.messages[0]")


def test_validate_extract_path_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="malformed|empty"):
        validate_extract_path("values..content")
    with pytest.raises(ValueError, match="unclosed"):
        validate_extract_path("values.messages[0")


def test_validate_extract_paths_enforces_max() -> None:
    too_many = {f"k{i}": f"values.f{i}" for i in range(11)}
    with pytest.raises(ValueError, match="at most 10"):
        validate_extract_paths(too_many)


def test_resolve_json_path_reads_nested_and_indexes() -> None:
    root = {
        "messages": [
            {"content": "first"},
            {"content": "second"},
        ]
    }
    assert resolve_json_path(root, "values.messages[0].content") == "first"
    assert resolve_json_path(root, "values.messages[-1].content") == "second"


def test_resolve_json_path_missing_returns_none() -> None:
    assert resolve_json_path({"messages": []}, "values.messages[0].content") is None
    assert resolve_json_path({}, "metadata.title") is None


def test_sources_needed_by_extract() -> None:
    assert sources_needed_by_extract(
        {"a": "values.x", "b": "metadata.y", "c": "config.z"}
    ) == {"values", "metadata", "config"}


def test_thread_search_request_rejects_unknown_select() -> None:
    with pytest.raises(ValidationError):
        ThreadSearchRequest(select=["thread_id", "not_a_field"])


def test_thread_search_request_rejects_bad_extract() -> None:
    with pytest.raises(ValidationError):
        ThreadSearchRequest(extract={"t": "foo.bar"})


def test_thread_search_request_accepts_select_and_extract() -> None:
    req = ThreadSearchRequest(
        select=["thread_id", "values"],
        extract={"title": "values.messages[0].content"},
    )
    assert req.select == ["thread_id", "values"]
    assert req.extract == {"title": "values.messages[0].content"}
