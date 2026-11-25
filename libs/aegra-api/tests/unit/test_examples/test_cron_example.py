"""Regression tests for the cron example graph in examples/cron_example."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[5] / "examples"


@pytest.fixture
def cron_graph(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    yield importlib.import_module("cron_example.graph")
    for name in [name for name in sys.modules if name == "cron_example" or name.startswith("cron_example.")]:
        del sys.modules[name]


async def test_tick_records_the_cron_firing_behind_the_run(cron_graph: ModuleType) -> None:
    result = await cron_graph.graph.ainvoke(
        {"messages": []},
        config={"configurable": {"cron_id": "cron-1", "cron_scheduled_time": "2026-01-01T09:00:00+00:00"}},
    )

    tick = result["messages"][-1]
    assert tick.content.startswith("Tick #1 at ")
    assert tick.additional_kwargs["cron_id"] == "cron-1"
    assert tick.additional_kwargs["cron_scheduled_time"] == "2026-01-01T09:00:00+00:00"


async def test_tick_outside_a_cron_records_no_firing(cron_graph: ModuleType) -> None:
    result = await cron_graph.graph.ainvoke({"messages": []})

    tick = result["messages"][-1]
    assert tick.additional_kwargs["cron_id"] is None
    assert tick.additional_kwargs["cron_scheduled_time"] is None
