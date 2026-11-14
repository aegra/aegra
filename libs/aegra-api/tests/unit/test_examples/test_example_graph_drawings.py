"""The graph view Aegra serves shows every route in examples/stress_test and examples/react_agent_hitl."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[5] / "examples"


@pytest.fixture
def examples_on_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    yield
    for name in [name for name in sys.modules if name.split(".")[0] in {"stress_test", "react_agent_hitl"}]:
        del sys.modules[name]


def _edges_from(module_name: str, source: str) -> set[str]:
    graph = importlib.import_module(module_name).graph
    return {edge.target for edge in graph.get_graph().edges if edge.source == source}


def test_stress_test_graph_draws_its_loop_and_respond(examples_on_path: None) -> None:
    assert _edges_from("stress_test.graph", "process") == {"process", "respond"}


def test_hitl_graph_draws_every_human_approval_route(examples_on_path: None) -> None:
    assert _edges_from("react_agent_hitl.graph", "human_approval") == {"tools", "call_model", "__end__"}
