"""Run-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aegra_api.utils.status_compat import validate_run_status


class RunCreate(BaseModel):
    """Request model for creating runs"""

    assistant_id: str = Field(..., description="Assistant to execute")
    input: dict[str, Any] | None = Field(
        None,
        description="Input data for the run. Optional when resuming from a checkpoint.",
    )
    config: dict[str, Any] | None = Field(default_factory=dict, description="Execution config")
    context: dict[str, Any] | None = Field(default_factory=dict, description="Execution context")
    checkpoint: dict[str, Any] | None = Field(
        None,
        description="Checkpoint configuration (e.g., {'checkpoint_id': '...', 'checkpoint_ns': ''})",
    )
    stream: bool = Field(False, description="Enable streaming response")
    stream_mode: str | list[str] | None = Field(None, description="Requested stream mode(s)")
    on_disconnect: str | None = Field(
        None,
        description="Behavior on client disconnect: 'cancel' (default) or 'continue'.",
    )
    on_completion: Literal["delete", "keep"] | None = Field(
        None,
        description="Behavior after stateless run completes: 'delete' (default) removes the ephemeral thread, 'keep' preserves it.",
    )

    multitask_strategy: str | None = Field(
        None,
        description="Strategy for handling concurrent runs on same thread: 'reject', 'interrupt', 'rollback', or 'enqueue'.",
    )

    # Human-in-the-loop fields (core HITL functionality)
    command: dict[str, Any] | None = Field(
        None,
        description="Command for resuming interrupted runs with state updates or navigation",
    )
    interrupt_before: str | list[str] | None = Field(
        None,
        description="Nodes to interrupt immediately before they get executed. Use '*' for all nodes.",
    )
    interrupt_after: str | list[str] | None = Field(
        None,
        description="Nodes to interrupt immediately after they get executed. Use '*' for all nodes.",
    )

    # Subgraph configuration
    stream_subgraphs: bool | None = Field(
        False,
        description="Whether to include subgraph events in streaming. When True, includes events from all subgraphs. When False (default when None), excludes subgraph events. Defaults to False for backwards compatibility.",
    )

    # Request metadata (top-level in payload)
    metadata: dict[str, Any] | None = Field(
        None,
        description="Request metadata (e.g., from_studio flag)",
    )

    @model_validator(mode="after")
    def validate_input_command_exclusivity(self) -> Self:
        """Ensure input and command are mutually exclusive"""
        # Allow empty input dict when command is present (frontend compatibility)
        if self.input is not None and self.command is not None:
            # If input is just an empty dict, treat it as None for compatibility
            if self.input == {}:
                self.input = None
            else:
                raise ValueError("Cannot specify both 'input' and 'command' - they are mutually exclusive")
        if self.input is None and self.command is None:
            if self.checkpoint is not None:
                # Keep input as None so LangGraph resumes from the checkpoint.
                # Coercing to {} makes Pregel treat it as fresh input and restart
                # from __start__ instead of continuing from next=[...].
                pass
            else:
                raise ValueError("Must specify either 'input' or 'command'")
        return self


class Run(BaseModel):
    """Run entity model

    Status values: pending, running, error, success, timeout, interrupted
    """

    model_config = ConfigDict(from_attributes=True)

    run_id: str = Field(..., description="Unique identifier for the run.")
    thread_id: str = Field(..., description="Thread this run belongs to.")
    assistant_id: str = Field(..., description="Assistant that is executing this run.")
    status: str = Field(
        "pending", description="Current run status: pending, running, error, success, timeout, or interrupted."
    )
    input: dict[str, Any] = Field(..., description="Input data provided to the run.")
    output: dict[str, Any] | None = Field(
        None, description="Final output produced by the run, or null if not yet complete."
    )
    error_message: str | None = Field(None, description="Error message if the run failed.")
    config: dict[str, Any] | None = Field(
        default_factory=dict, description="Configuration passed to the graph at runtime."
    )
    context: dict[str, Any] | None = Field(
        default_factory=dict, description="Context variables available during execution."
    )
    user_id: str = Field(..., description="Identifier of the user who owns this run.")
    created_at: datetime = Field(..., description="Timestamp when the run was created.")
    updated_at: datetime = Field(..., description="Timestamp when the run was last updated.")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status conforms to API specification."""
        if not isinstance(v, str):
            raise ValueError(f"Status must be a string, got {type(v)}")
        return validate_run_status(v)


class RunStatus(BaseModel):
    """Simple run status response"""

    run_id: str = Field(..., description="Unique identifier for the run.")
    status: str = Field(..., description="Current run status value.")

    message: str | None = Field(None, description="Optional human-readable status message.")
