"""Job specification for graph run execution.

Groups the 15+ parameters of execute_run into three logical sub-objects
so every function in the execution chain takes at most three arguments.

Uses Pydantic models (consistent with the rest of the codebase) with
frozen=True for immutability. Serialization uses model_dump/model_validate
— no manual to_dict/from_dict needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from aegra_api.models.auth import User

if TYPE_CHECKING:
    from aegra_api.core.orm import Run as RunORM


class RunIdentity(BaseModel):
    """Locates a run within the system."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    thread_id: str
    graph_id: str


class RunExecution(BaseModel):
    """What to execute and how to execute it."""

    model_config = ConfigDict(frozen=True)

    input_data: dict[str, Any] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    stream_mode: str | list[str] | None = None
    checkpoint: dict[str, Any] | None = None
    command: dict[str, Any] | None = None


class RunBehavior(BaseModel):
    """Behavioral modifiers for the run."""

    model_config = ConfigDict(frozen=True)

    interrupt_before: str | list[str] | None = None
    interrupt_after: str | list[str] | None = None
    multitask_strategy: str | None = None
    subgraphs: bool = False


class RunJob(BaseModel):
    """Complete specification for executing a graph run.

    Serialization for DB storage and Redis transport uses Pydantic's
    built-in model_dump/model_validate — no manual methods needed.
    """

    model_config = ConfigDict(frozen=True)

    identity: RunIdentity
    user: User
    execution: RunExecution = RunExecution()
    behavior: RunBehavior = RunBehavior()

    def to_execution_params(self) -> dict[str, Any]:
        """Serialize for the ``execution_params`` JSONB column.

        Stores everything a worker needs to reconstruct the RunJob from
        the database. Identity fields (run_id, thread_id) are already
        columns on the runs table, but graph_id is not — so we include it.
        """
        return {
            "graph_id": self.identity.graph_id,
            "user": self.user.model_dump(),
            "execution": self.execution.model_dump(),
            "behavior": self.behavior.model_dump(),
        }

    @classmethod
    def from_run_orm(cls, run_orm: RunORM) -> RunJob:
        """Reconstruct a RunJob from a Run ORM row with execution_params."""
        params = run_orm.execution_params
        if params is None:
            raise ValueError(f"Run {run_orm.run_id} has no execution_params")
        return cls(
            identity=RunIdentity(
                run_id=run_orm.run_id,
                thread_id=run_orm.thread_id,
                graph_id=params["graph_id"],
            ),
            user=User.model_validate(params["user"]),
            execution=RunExecution.model_validate(params["execution"]),
            behavior=RunBehavior.model_validate(params["behavior"]),
        )
