"""The published schema for AssistantCreate describes the deployment it came from.

Whether `graph_id` may be omitted depends on `aegra.json`, so the generated
schema — and every client generated from it — has to say which case this
deployment is in.
"""

import json
from pathlib import Path

import pytest

from aegra_api.models.assistants import AssistantCreate


def _use_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "aegra.json").write_text(json.dumps(config))


def test_schema_advertises_the_resolved_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolvable default is optional, and clients are told which graph it is."""
    _use_config(
        tmp_path,
        monkeypatch,
        {
            "default_graph_id": "other",
            "graphs": {"agent": "./agent.py:graph", "other": "./other.py:graph"},
        },
    )

    schema = AssistantCreate.model_json_schema()

    assert schema["properties"]["graph_id"]["default"] == "other"
    assert "graph_id" not in schema.get("required", [])


def test_schema_advertises_the_sole_graph_as_the_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The single-graph deployment needs no config to publish its default."""
    _use_config(tmp_path, monkeypatch, {"graphs": {"agent": "./agent.py:graph"}})

    schema = AssistantCreate.model_json_schema()

    assert schema["properties"]["graph_id"]["default"] == "agent"
    assert "graph_id" not in schema.get("required", [])


def test_schema_keeps_graph_id_required_without_a_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Several graphs and no nominated default: clients must still send one."""
    _use_config(
        tmp_path,
        monkeypatch,
        {"graphs": {"agent": "./agent.py:graph", "other": "./other.py:graph"}},
    )

    schema = AssistantCreate.model_json_schema()

    assert "graph_id" in schema["required"]
    assert "default" not in schema["properties"]["graph_id"]


def test_schema_without_a_default_does_not_permit_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Required and non-nullable go together, because a null is refused.

    The model's field is nullable because a resolved default can stand in for
    it. Where nothing resolves, a null is treated as omitted and answered 422,
    so publishing the nullable branch would describe a request the server
    refuses as schema-valid.
    """
    _use_config(
        tmp_path,
        monkeypatch,
        {"graphs": {"agent": "./agent.py:graph", "other": "./other.py:graph"}},
    )

    graph_id = AssistantCreate.model_json_schema()["properties"]["graph_id"]

    assert graph_id["type"] == "string"
    assert "anyOf" not in graph_id


def test_schema_with_a_default_still_permits_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where a default resolves, a null means "use it", so it stays valid."""
    _use_config(tmp_path, monkeypatch, {"graphs": {"agent": "./agent.py:graph"}})

    graph_id = AssistantCreate.model_json_schema()["properties"]["graph_id"]

    assert {"type": "null"} in graph_id["anyOf"]
