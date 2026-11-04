"""Status enums for Aegra API specification."""

from typing import Literal

# Run status enum
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

RunCancellationAction = Literal["cancel", "interrupt"]

# Why the worker cancelled a run's task, carried as the message of the CancelledError
# that unwinds the run: a graph reads it to tell a hand-off from a cancel that ends the run.
CancelReason = Literal["cancel", "requeue", "lease_lost", "timeout"]
