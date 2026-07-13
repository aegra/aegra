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
from aegra_api.models.webhooks import WebhookConfig

if TYPE_CHECKING:
    from aegra_api.core.orm import Run as RunORM


class RunIdentity(BaseModel):
    """Locates a run within the system."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    thread_id: str
    graph_id: str
    # Carried so the completion webhook payload can report assistant_id.
    assistant_id: str | None = None


class RunExecution(BaseModel):
    """What to execute and how to execute it."""

    model_config = ConfigDict(frozen=True)

    input_data: dict[str, Any] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    stream_mode: str | list[str] | None = None
    checkpoint: dict[str, Any] | None = None
    command: dict[str, Any] | None = None
    # When true, stream via the native v3 protocol producer for Agent Protocol v2.
    event_streaming_v2: bool = False
    # POSTed on run completion; round-trips through execution_params in both executors.
    # Its secret/headers sit unmasked there, but execution_params has no client read path.
    webhook: WebhookConfig | None = None


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
    # JSONB-persisted; deliberately ``dict[str, Any]`` so legacy/drifted rows stay readable via
    # ``from_run_orm``. OTEL boundary in ``merge_run_metadata`` filters at emit. Do not narrow.
    run_metadata: dict[str, Any] = Field(default_factory=dict)

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
            "run_metadata": self.run_metadata,
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
                # getattr tolerates lightweight fake ORMs in tests; real rows always have it.
                assistant_id=getattr(run_orm, "assistant_id", None),
            ),
            user=User.model_validate(params["user"]),
            execution=RunExecution.model_validate(params["execution"]),
            behavior=RunBehavior.model_validate(params["behavior"]),
            run_metadata=params.get("run_metadata") or {},
        )
