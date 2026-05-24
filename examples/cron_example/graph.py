"""Cron example graph — demonstrates scheduled execution with aegra crons.

This graph simulates a periodic task: it records the current time and
a counter of how many times it has run (derived from the message history).

Schedule examples:
  "* * * * *"     — every minute
  "*/5 * * * *"   — every 5 minutes
  "0 * * * *"     — every hour
  "0 9 * * *"     — every day at 9:00 UTC

To try it:
  1. Add "cron_example" to aegra.json graphs section.
  2. Create an assistant: POST /assistants {"graph_id": "cron_example"}
  3. Create a cron:     POST /runs/crons {
       "assistant_id": "<id>",
       "schedule": "* * * * *",
       "thread_id": "<optional — keep a persistent thread across runs>"
     }
  4. Each scheduled run appends a new "tick" message to the thread.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import StateGraph, add_messages


@dataclass
class State:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)


def tick(state: State) -> dict[str, Any]:
    """Record a timestamped tick and the total run count."""
    run_number = len(state.messages) + 1
    now = datetime.now(UTC).isoformat()
    return {
        "messages": [
            AIMessage(
                content=f"Tick #{run_number} at {now}",
                additional_kwargs={"run_number": run_number, "timestamp": now},
            )
        ]
    }


builder = StateGraph(State)
builder.add_node("tick", tick)
builder.set_entry_point("tick")
builder.set_finish_point("tick")

graph = builder.compile()
