"""The react-agent template ends a run with its step-limit message at any recursion limit."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from aegra_cli.cli import cli
from aegra_cli.templates import slugify


class AlwaysCallsTool:
    """Requests the `add` tool on every call, so only the step limit can end the loop."""

    def __init__(self, message_class: type[Any]) -> None:
        self.message_class = message_class
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "AlwaysCallsTool":
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        tool_call = {"name": "add", "args": {"a": 1, "b": 2}, "id": f"call_{self.calls}"}
        return self.message_class(content="", tool_calls=[tool_call])


@pytest.fixture
def react_project(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    # The rendered utils.py imports langchain, which is only in aegra-api's dev group
    pytest.importorskip("langchain")
    project_dir = tmp_path / "step-limit"
    result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "2"])
    assert result.exit_code == 0
    slug = slugify("step-limit")
    monkeypatch.syspath_prepend(str(project_dir / "src"))
    yield slug
    for name in [name for name in sys.modules if name.split(".")[0] == slug]:
        del sys.modules[name]


@pytest.mark.asyncio
@pytest.mark.parametrize("recursion_limit", [3, 4, 5, 6, 25])
async def test_template_returns_the_step_limit_message_at_any_recursion_limit(
    react_project: str, monkeypatch: pytest.MonkeyPatch, recursion_limit: int
) -> None:
    graph_module = importlib.import_module(f"{react_project}.graph")
    context_module = importlib.import_module(f"{react_project}.context")
    model = AlwaysCallsTool(graph_module.AIMessage)

    def load_chat_model(fully_specified_name: str) -> AlwaysCallsTool:
        return model

    monkeypatch.setattr(graph_module, "load_chat_model", load_chat_model)

    result = await graph_module.graph.ainvoke(
        {"messages": [{"role": "user", "content": "go"}]},
        {"recursion_limit": recursion_limit},
        context=context_module.Context(),
    )

    assert result["messages"][-1].content == (
        "Sorry, I could not find an answer to your question in the specified number of steps."
    )
