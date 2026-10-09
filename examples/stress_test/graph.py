"""Stress test graph — no LLM calls, configurable delay, deterministic output.

Used by multi-instance stress tests to validate worker architecture
at high concurrency without burning API tokens.

Input:
  {"messages": [{"role": "user", "content": "..."}]}
  Content can optionally be JSON:
  {"delay": 2.0, "steps": 3, "fail": false, "interrupt": false, "_m": "marker"}
  ("interrupt": true pauses on a HITL interrupt(); "_m" tags the echo so tests
  can assert which runs executed and in what order.)
  Optional "_cancellation_probe" supplies a scoped store namespace, label, and
  hold_cancellation flag for the pre-emption E2E tests.

Output:
  Echoes back with metadata about execution (node count, total delay).
"""

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.config import get_store
from langgraph.graph import StateGraph, add_messages
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

logger = logging.getLogger(__name__)


@dataclass
class State:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    step_count: int = 0
    total_delay: float = 0.0


def _parse_config(state: State) -> dict[str, Any]:
    """Extract delay/steps/fail config from the last user message."""
    defaults = {"delay": 0.5, "steps": 2, "fail": False}
    if not state.messages:
        return defaults

    content = state.messages[-1].content
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {**defaults, **parsed}
    except (json.JSONDecodeError, TypeError):
        pass
    return defaults


async def _hold_cancellation(store: BaseStore, namespace: tuple[str, ...]) -> None:
    deadline = asyncio.get_running_loop().time() + 60
    while asyncio.get_running_loop().time() < deadline:
        try:
            released = await store.aget(namespace, "release")
            if released is not None:
                return
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # Heartbeat lease loss can cancel twice; the caller re-raises after release.
            logger.info("Deferring repeated cancellation until the test releases the probe")
    raise TimeoutError("Cancellation probe was not released within 60 seconds")


async def _simulate_work(delay: float, probe: dict[str, Any] | None) -> None:
    if probe is None:
        await asyncio.sleep(delay)
        return

    store = get_store()
    namespace = tuple(probe["namespace"])
    label = probe["label"]
    predecessor_exited = await store.aget(namespace, "A.exited")
    await store.aput(namespace, f"{label}.started", {"predecessor_exited": predecessor_exited is not None})
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        await store.aput(namespace, f"{label}.cancelling", {"seen": True})
        if probe.get("hold_cancellation"):
            await _hold_cancellation(store, namespace)
        raise
    finally:
        await store.aput(namespace, f"{label}.exited", {"seen": True})


async def process_step(state: State) -> dict[str, Any]:
    """Simulate work with a configurable delay."""
    config = _parse_config(state)
    delay = float(config["delay"])

    await _simulate_work(delay, config.get("_cancellation_probe"))

    # Optional human-in-the-loop pause (LLM-free HITL fixture), AFTER the simulated work so a
    # caller can enqueue behind a still-running run before it pauses. Stays paused until resumed.
    if config.get("interrupt"):
        interrupt({"awaiting": config.get("_m", "input")})

    new_step = state.step_count + 1
    new_delay = state.total_delay + delay

    if config.get("fail") and new_step >= int(config.get("steps", 2)):
        raise RuntimeError(f"Intentional failure at step {new_step}")

    return {
        "step_count": new_step,
        "total_delay": new_delay,
    }


async def respond(state: State) -> dict[str, Any]:
    """Generate a deterministic response with execution metadata."""
    return {
        "messages": [
            AIMessage(
                content=json.dumps(
                    {
                        "echo": state.messages[-1].content if state.messages else "",
                        "steps_completed": state.step_count,
                        "total_delay_seconds": round(state.total_delay, 2),
                        "status": "completed",
                    }
                )
            )
        ],
    }


def should_continue(state: State) -> str:
    """Loop through process_step until reaching the configured step count."""
    config = _parse_config(state)
    target_steps = int(config.get("steps", 2))
    if state.step_count < target_steps:
        return "process"
    return "respond"


builder = StateGraph(State)
builder.add_node("process", process_step)
builder.add_node("respond", respond)
builder.set_entry_point("process")
builder.add_conditional_edges("process", should_continue)
builder.set_finish_point("respond")

graph = builder.compile()
