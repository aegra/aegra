"""Tests for the cron example graph in examples/cron_example."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[5] / "examples"


@pytest.fixture
def cron_graph(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    return importlib.import_module("cron_example.graph")


@pytest.mark.asyncio
async def test_tick_records_the_display_name_the_run_ran_as(cron_graph: ModuleType) -> None:
    result = await cron_graph.graph.ainvoke({"messages": []}, {"configurable": {"user_display_name": "cron-scheduler"}})

    tick = result["messages"][-1]
    assert tick.content.startswith("Tick #1 at ")
    assert tick.additional_kwargs["run_by"] == "cron-scheduler"


@pytest.mark.asyncio
async def test_tick_records_no_display_name_outside_the_server(cron_graph: ModuleType) -> None:
    result = await cron_graph.graph.ainvoke({"messages": []})

    assert result["messages"][-1].additional_kwargs["run_by"] is None
