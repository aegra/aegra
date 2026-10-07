"""Stress test agent with slow tool calls.

An LLM-powered ReAct agent that must call a slow_process tool multiple times
to complete a task. Each tool call takes ~2 seconds, simulating real-world
API calls, database queries, or external service latency.

Used for stress testing worker architecture under realistic load.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from react_agent.utils import load_chat_model


@dataclass
class State:
    """Agent state with message history."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    remaining_steps: RemainingSteps = 25


@tool
async def slow_process(step_number: int) -> str:
    """Process a step in the multi-step pipeline. Each step takes about 2 seconds.

    Args:
        step_number: The step number to process (1-10).

    Returns:
        A confirmation message for the completed step.
    """
    await asyncio.sleep(2)
    return f"Step {step_number} completed successfully. Result: data_chunk_{step_number}"


TOOLS = [slow_process]


async def call_model(state: State) -> dict[str, list[AIMessage]]:
    """Call the LLM with tool binding."""
    model = load_chat_model("openai/gpt-4o-mini").bind_tools(TOOLS)

    system_message = (
        "You are a data processing agent. When asked to process data, you MUST call "
        "the slow_process tool for each step sequentially from step 1 to the number "
        "requested. Call one step at a time, wait for the result, then call the next. "
        "After all steps are done, summarize the results briefly."
    )

    response = await model.ainvoke(
        [{"role": "system", "content": system_message}, *state.messages],
    )

    if state.remaining_steps < 3 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Processing incomplete — reached step limit.",
                )
            ]
        }

    return {"messages": [response]}


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Route to tools if the model wants to call one, otherwise end."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")
    if not last_message.tool_calls:
        return "__end__"
    return "tools"


builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")

graph = builder.compile(name="Stress Tool Agent")
