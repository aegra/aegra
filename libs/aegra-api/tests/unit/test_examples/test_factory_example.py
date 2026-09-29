"""Regression tests for the factory example graph in examples/factory."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

EXAMPLES_DIR = Path(__file__).resolve().parents[5] / "examples"


class ScriptedModel:
    """Asks for search_web on its first call when tools are bound, then answers."""

    def __init__(self) -> None:
        self.received: list[list[BaseMessage]] = []
        self.tools_bound = False

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        self.tools_bound = True
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.received.append(list(messages))
        if self.tools_bound and len(self.received) == 1:
            return AIMessage(
                content="", tool_calls=[{"name": "search_web", "args": {"query": "aegra"}, "id": "call_1"}]
            )
        return AIMessage(content=f"answer {len(self.received)}")


@pytest.fixture
def factory_graph(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    yield importlib.import_module("factory.graph")
    for name in [name for name in sys.modules if name == "factory" or name.startswith("factory.")]:
        del sys.modules[name]


def _use_model(monkeypatch: pytest.MonkeyPatch, factory_graph: ModuleType, model: ScriptedModel) -> None:
    def load_chat_model(fully_specified_name: str) -> ScriptedModel:
        return model

    monkeypatch.setattr(factory_graph, "load_chat_model", load_chat_model)


async def test_model_sees_question_and_tool_call_after_a_tool_round(
    monkeypatch: pytest.MonkeyPatch, factory_graph: ModuleType
) -> None:
    model = ScriptedModel()
    _use_model(monkeypatch, factory_graph, model)
    tools_module = importlib.import_module("factory.tools")
    context_module = importlib.import_module("factory.context")
    graph = factory_graph._build_graph([tools_module.search_web])

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is Aegra?")]}, context=context_module.FactoryContext()
    )

    assert [type(message) for message in model.received[1]] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
    assert [type(message) for message in result["messages"]] == [HumanMessage, AIMessage, ToolMessage, AIMessage]


async def test_thread_keeps_earlier_turns(monkeypatch: pytest.MonkeyPatch, factory_graph: ModuleType) -> None:
    model = ScriptedModel()
    _use_model(monkeypatch, factory_graph, model)
    context_module = importlib.import_module("factory.context")
    graph = factory_graph._build_graph([]).copy(update={"checkpointer": InMemorySaver()})
    config = {"configurable": {"thread_id": "factory-history"}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="My name is Ada.")]}, config, context=context_module.FactoryContext()
    )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="What is my name?")]}, config, context=context_module.FactoryContext()
    )

    assert [message.content for message in model.received[1][1:]] == ["My name is Ada.", "answer 1", "What is my name?"]
