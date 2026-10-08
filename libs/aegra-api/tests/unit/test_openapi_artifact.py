"""The checked-in spec must not carry one deployment's graph configuration.

`docs/openapi.json` is the published reference, not a snapshot of whoever ran
`make openapi` — which is why the exporter already normalizes the title and
version. `graph_id` needs the same treatment: a contributor whose `aegra.json`
resolves a default would otherwise regenerate the file with `graph_id` optional
and quietly relax the published contract for everyone.

A deployment that resolves a default still serves its own schema with
`graph_id` optional. That is correct for that server; this file states what
every server accepts.
"""

import json
from pathlib import Path
from typing import Any

import pytest

_SPEC = Path(__file__).resolve().parents[4] / "docs" / "openapi.json"


@pytest.fixture(scope="module")
def assistant_create() -> dict[str, Any]:
    return json.loads(_SPEC.read_text())["components"]["schemas"]["AssistantCreate"]


def test_graph_id_is_published_as_required(assistant_create: dict[str, Any]) -> None:
    """An explicit graph_id is accepted by every deployment; omitting one is not."""
    assert "graph_id" in assistant_create["required"]


def test_graph_id_is_published_as_a_plain_string(assistant_create: dict[str, Any]) -> None:
    """Required and nullable together would describe {"graph_id": null} as valid."""
    graph_id = assistant_create["properties"]["graph_id"]

    assert graph_id.get("type") == "string"
    assert "anyOf" not in graph_id


def test_no_deployment_default_leaks_into_the_reference(assistant_create: dict[str, Any]) -> None:
    """A `default` here would name the graph of whoever last ran the exporter."""
    assert "default" not in assistant_create["properties"]["graph_id"]
