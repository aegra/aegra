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
