"""The live_input example graph declares live input and adds what it takes before each step."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from aegra_api.services.injected_input import INJECTED_INPUT_KEY, InjectedMessage

EXAMPLES_DIR = Path(__file__).resolve().parents[5] / "examples"


class FakeReader:
    """Stands in for the server's reader: queued messages appear on the next pending()."""

    def __init__(self, *queued: InjectedMessage) -> None:
        self.queued = list(queued)
        self.accepted = 0
        self.taken: list[str] = []

    async def accept(self) -> None:
        self.accepted += 1

    async def pending(self) -> list[InjectedMessage]:
        return [item for item in self.queued if item.message_id not in self.taken]

    async def take(self, message_ids: list[str]) -> list[str]:
        self.taken.extend(message_ids)
        return message_ids


def _injected(message_id: str, content: str) -> InjectedMessage:
    message = {"role": "user", "content": content, "id": message_id, "additional_kwargs": {"injected_by": "u1"}}
    return InjectedMessage(message_id=message_id, message=message)


@pytest.fixture
def graph_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    yield importlib.import_module("live_input.graph")
    for name in [name for name in sys.modules if name.split(".")[0] == "live_input"]:
        del sys.modules[name]


def test_example_reads_the_key_the_server_pins(graph_module: Any) -> None:
    assert graph_module.INJECTED_INPUT_KEY == INJECTED_INPUT_KEY


async def test_injected_messages_enter_the_conversation_before_the_step_reply(graph_module: Any) -> None:
    reader = FakeReader(_injected("m1", "also check the tests"))

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="start")], "steps": 2, "delay": 0},
        {"configurable": {INJECTED_INPUT_KEY: reader}},
    )

    assert reader.accepted == 1
    assert reader.taken == ["m1"]
    messages = result["messages"]
    assert [type(message).__name__ for message in messages] == [
        "HumanMessage",
        "HumanMessage",
        "AIMessage",
        "AIMessage",
    ]
    assert messages[1].id == "m1"
    assert messages[1].additional_kwargs == {"injected_by": "u1"}
    assert messages[2].content == "step 1: received ['also check the tests']"
    assert messages[3].content == "step 2: received []"


async def test_drain_false_declares_but_takes_nothing(graph_module: Any) -> None:
    reader = FakeReader(_injected("m1", "later"))

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="start")], "steps": 1, "delay": 0, "drain": False},
        {"configurable": {INJECTED_INPUT_KEY: reader}},
    )

    assert reader.accepted == 1
    assert reader.taken == []
    assert [message.content for message in result["messages"] if isinstance(message, AIMessage)] == [
        "step 1: received []"
    ]


async def test_runs_without_a_reader_outside_the_server(graph_module: Any) -> None:
    result = await graph_module.graph.ainvoke({"messages": [HumanMessage(content="start")], "steps": 1, "delay": 0})

    assert result["messages"][-1].content == "step 1: received []"


async def test_a_message_the_state_already_holds_is_taken_but_not_added_again(graph_module: Any) -> None:
    """After an interrupt the resume sees again what the paused run took and added."""
    reader = FakeReader(_injected("m1", "also check the tests"))
    held = HumanMessage(content="also check the tests", id="m1", additional_kwargs={"injected_by": "u1"})

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="start"), held], "steps": 1, "delay": 0},
        {"configurable": {INJECTED_INPUT_KEY: reader}},
    )

    assert reader.taken == ["m1"]
    assert [message.id for message in result["messages"]].count("m1") == 1
    assert result["messages"][-1].content == "step 1: received []"


async def test_pause_after_interrupts_once_and_declares_again_on_resume(graph_module: Any) -> None:
    reader = FakeReader()
    graph = graph_module.builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {INJECTED_INPUT_KEY: reader, "thread_id": "t1"}}

    paused = await graph.ainvoke(
        {"messages": [HumanMessage(content="start")], "steps": 2, "delay": 0, "pause_after": 1}, config
    )
    assert [item.value for item in paused["__interrupt__"]] == ["continue?"]
    assert reader.accepted == 2

    result = await graph.ainvoke(Command(resume="yes"), config)

    assert reader.accepted == 3
    assert [message.content for message in result["messages"] if isinstance(message, AIMessage)] == [
        "step 1: received []",
        "step 2: received []",
    ]
