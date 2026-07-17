"""Status enums for Aegra API specification."""

from typing import Literal

# Run status enum (API wire vocabulary — matches the LangGraph SDK). The internal
# double-texting park state 'queued' is persisted-only and reported as 'pending'.
RunStatus = Literal[
    "pending",
    "running",
    "error",
    "success",
    "timeout",
    "interrupted",
]

# Thread status enum
ThreadStatus = Literal[
    "idle",
    "busy",
    "interrupted",
    "error",
]

# Multitask strategy enum
MultitaskStrategy = Literal[
    "reject",
    "rollback",
    "interrupt",
    "enqueue",
]

# Applied when a run is created without an explicit multitask_strategy.
# Matches LangGraph Platform's documented default (enqueue).
MULTITASK_DEFAULT: MultitaskStrategy = "enqueue"
