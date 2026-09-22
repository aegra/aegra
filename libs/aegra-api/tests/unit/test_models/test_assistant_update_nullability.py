"""What ``AssistantUpdate`` publishes must match what the endpoint accepts.

``name`` and ``graph_id`` back NOT NULL columns, and ``update_assistant``
answers a supplied empty one with 422 rather than falling back to the stored
value, because a caller who supplied a field meant to change it. Publishing
those two as nullable described ``{"name": null}`` as valid and would have a
generated client offer the one value that cannot be stored.

They stay nullable on the model so a null arrives as a *supplied* value and
gets that 422, rather than a generic type error that does not say to omit the
field instead. The other four are genuinely nullable: a null clears
``description``, and ``config``, ``context`` and ``metadata`` read it as empty.
"""

from typing import Any

import pytest

from aegra_api.models.assistants import AssistantUpdate

_SCHEMA: dict[str, Any] = AssistantUpdate.model_json_schema()

_NOT_NULL_COLUMNS = ["name", "graph_id"]
_NULLABLE = ["description", "config", "context", "metadata"]


@pytest.mark.parametrize("field", _NOT_NULL_COLUMNS)
def test_a_field_backing_a_not_null_column_is_published_non_nullable(field: str) -> None:
    published = _SCHEMA["properties"][field]

    assert published.get("type") == "string"
    assert "anyOf" not in published


@pytest.mark.parametrize("field", _NOT_NULL_COLUMNS)
def test_no_sentinel_default_is_published(field: str) -> None:
    """``"default": null`` under ``"type": "string"`` is the same null by another name."""
    assert "default" not in _SCHEMA["properties"][field]


@pytest.mark.parametrize("field", _NOT_NULL_COLUMNS)
def test_omission_is_still_how_a_field_is_left_unchanged(field: str) -> None:
    """Non-nullable must not mean required."""
    assert field not in _SCHEMA.get("required", [])
    assert AssistantUpdate().model_dump(exclude_unset=True) == {}


@pytest.mark.parametrize("field", _NOT_NULL_COLUMNS)
def test_a_null_is_carried_to_the_service_as_supplied(field: str) -> None:
    """The service distinguishes a supplied null from an omission; so must the model.

    Rejecting the null here would be indistinguishable from a type error and
    would lose the answer that tells the caller to omit the field instead.
    """
    request = AssistantUpdate.model_validate({field: None})

    assert request.model_dump(exclude_unset=True) == {field: None}


@pytest.mark.parametrize("field", _NULLABLE)
def test_a_field_the_service_reads_as_empty_stays_published_nullable(field: str) -> None:
    published = _SCHEMA["properties"][field]

    assert "anyOf" in published
    assert {"type": "null"} in published["anyOf"]
