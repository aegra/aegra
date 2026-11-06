"""Live input example: a graph that takes messages sent with ``input.inject`` while it runs.

No LLM calls. The run declares at its start that it takes live input, then works
in ``steps`` steps of ``delay`` seconds. Before each step it takes the messages
injected since the last one and adds them to the conversation, where a model
call would see them.

Input:
  {"messages": [...], "steps": 3, "delay": 0.5, "drain": true, "pause_after": 0}

``drain: false`` declares live input but takes nothing, so injected messages
wait for the thread's next run. ``pause_after: N`` pauses for approval once,
after step N; the resume is a new run that continues at the pause.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import interrupt

# Aegra pins the run's reader under this configurable key; it is absent outside a server run.
INJECTED_INPUT_KEY = "aegra_injected_input"


@dataclass
class State:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    steps: int = 3
    delay: float = 0.5
    drain: bool = True
    pause_after: int = 0
    step_count: int = 0
    approved: bool = False


def _reader(config: RunnableConfig) -> Any | None:
    return config.get("configurable", {}).get(INJECTED_INPUT_KEY)


async def accept(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Declare that this run takes input.inject; without it the server refuses injects."""
    reader = _reader(config)
    if reader is not None:
        await reader.accept()
    return {"step_count": 0}


async def work(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Take the messages injected so far, then do one step of work."""
    injected: list[dict[str, Any]] = []
    reader = _reader(config)
    if reader is not None and state.drain:
        pending = await reader.pending()
        # take() reports what this node claimed; another run, or an earlier node of this run, may hold the rest.
        taken = set(await reader.take([item.message_id for item in pending]))
        # A resume sees again what the paused run took: taking it settles it, and the state already holds it.
        held = {message.id for message in state.messages}
        injected = [item.message for item in pending if item.message_id in taken and item.message_id not in held]

    await asyncio.sleep(state.delay)
    step = state.step_count + 1
    seen = [message["content"] for message in injected]
    return {"messages": [*injected, AIMessage(content=f"step {step}: received {seen}")], "step_count": step}


async def approve(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Pause for approval; the resumed run starts here, so it declares live input again."""
    reader = _reader(config)
    if reader is not None:
        await reader.accept()
    interrupt("continue?")
    return {"approved": True}


def route(state: State) -> str:
    if state.pause_after and state.step_count == state.pause_after and not state.approved:
        return "approve"
    return "work" if state.step_count < state.steps else END


builder = StateGraph(State)
builder.add_node("accept", accept)
builder.add_node("work", work)
builder.add_node("approve", approve)
builder.add_edge(START, "accept")
builder.add_edge("accept", "work")
builder.add_conditional_edges("work", route)
builder.add_conditional_edges("approve", route)

graph = builder.compile()
