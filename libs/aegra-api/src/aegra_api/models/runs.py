"""Run-related Pydantic models for Agent Protocol"""

import re
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aegra_api.models.webhooks import WebhookField
from aegra_api.utils.status_compat import validate_run_status

# Constraints for ``RunCreate.metadata`` keys/values, enforced at request
# time so the OpenAPI schema is honest about what reaches OTEL.  Without
# these limits a tenant could submit thousands of keys, megabyte-scale
# values, or nested structures — all of which would either be silently
# dropped by ``merge_run_metadata`` or balloon span size past the OTEL
# collector limits.  Bounds chosen to be generous for legitimate use
# (tenant id, feature flag, environment, sub-agent type, ...) while
# closing the DoS surface.
_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_METADATA_MAX_KEYS = 32
_METADATA_MAX_VALUE_LEN = 512


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

    # Accepts a bare URL string (SDK contract) or a rich webhook object.
    webhook: WebhookField = Field(None, description="URL or webhook object POSTed when the run completes.")

    # Request metadata (top-level in payload).  Reaches OTEL trace
    # attributes as ``langfuse.trace.metadata.<key>`` (and the
    # OpenInference ``metadata.<key>`` alias on Phoenix targets).  The
    # field is annotated ``dict[str, Any]`` rather than a primitive
    # union so a malformed payload produces one actionable 422 message
    # from ``validate_metadata_shape`` instead of N parallel union-arm
    # errors (one per primitive type Pydantic tries) per offending key.
    metadata: dict[str, Any] | None = Field(
        None,
        description=(
            "Request metadata propagated to OTEL trace attributes "
            "(``langfuse.trace.metadata.<key>``).  Keys must match "
            "``[A-Za-z0-9_-]{1,64}``.  Values must be primitive "
            "(``str``, ``int``, ``float``, ``bool``); string values are "
            "capped at 512 characters.  Maximum 32 keys.  Use this for "
            "filterable attributes (tenant, feature flag, environment, "
            "sub-agent type) rather than payload data."
        ),
    )

    @field_validator("metadata", mode="after")
    @classmethod
    def validate_metadata_shape(
        cls,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Enforce key shape, key count, value type, and string-value length.

        Validation runs entirely here (rather than relying on a primitive
        union on the field type) so each violation produces one clear
        error message instead of N parallel union-arm errors per offending
        key — easier for clients to surface to humans.
        """
        if metadata is None:
            return None
        if len(metadata) > _METADATA_MAX_KEYS:
            raise ValueError(f"metadata exceeds {_METADATA_MAX_KEYS} keys (got {len(metadata)})")
        for key, value in metadata.items():
            if not _METADATA_KEY_RE.match(key):
                raise ValueError(f"metadata key {key!r} must match {_METADATA_KEY_RE.pattern}")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"metadata value for key {key!r} must be str/int/float/bool, got {type(value).__name__}"
                )
            if isinstance(value, str) and len(value) > _METADATA_MAX_VALUE_LEN:
                raise ValueError(f"metadata value for key {key!r} exceeds {_METADATA_MAX_VALUE_LEN} characters")
        return metadata

    @model_validator(mode="after")
    def validate_input_command_exclusivity(self) -> Self:
        """Ensure input and command are mutually exclusive."""
        # Empty input dict alongside command: drop it for frontend compatibility.
        if self.input is not None and self.command is not None:
            if self.input == {}:
                self.input = None
            else:
                raise ValueError("Cannot specify both 'input' and 'command' - they are mutually exclusive")
        # Checkpoint-only resume keeps input=None so Pregel resumes from next=[...]
        # instead of restarting from __start__ with an empty input.
        if self.input is None and self.command is None and self.checkpoint is None:
            raise ValueError("Must specify at least one of 'input', 'command', or 'checkpoint'")
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
    input: dict[str, Any] | None = Field(
        None, description="Input data provided to the run. None for checkpoint-only resume."
    )
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
