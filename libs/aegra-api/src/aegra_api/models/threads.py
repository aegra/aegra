"""Thread-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aegra_api.utils.status_compat import validate_thread_status


class ThreadCreate(BaseModel):
    """Request model for creating threads"""

    model_config = ConfigDict(populate_by_name=True)

    metadata: dict[str, Any] | None = Field(None, description="Thread metadata")
    initial_state: dict[str, Any] | None = Field(None, description="LangGraph initial state")
    thread_id: str | None = Field(
        None,
        alias="threadId",
        description="Optional client-provided thread ID for idempotent creation",
    )
    if_exists: str | None = Field(
        "raise",
        alias="ifExists",
        description="Behavior when thread exists: 'raise' (default) or 'do_nothing'",
    )


class ThreadUpdate(BaseModel):
    """Request model for updating threads"""

    metadata: dict[str, Any] | None = Field(None, description="Thread metadata to update")


class Thread(BaseModel):
    """Thread entity model

    Status values: idle, busy, interrupted, error
    """

    model_config = ConfigDict(from_attributes=True)

    thread_id: str = Field(..., description="Unique identifier for the thread.")
    status: str = Field("idle", description="Current thread status: idle, busy, interrupted, or error.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attached to the thread.")
    user_id: str = Field(..., description="Identifier of the user who owns this thread.")
    created_at: datetime = Field(..., description="Timestamp when the thread was created.")
    updated_at: datetime = Field(..., description="Timestamp when the thread was last updated.")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status conforms to API specification."""
        if not isinstance(v, str):
            raise ValueError(f"Status must be a string, got {type(v)}")
        return validate_thread_status(v)


class ThreadList(BaseModel):
    """Response model for listing threads"""

    threads: list[Thread]
    total: int


THREAD_SEARCH_SELECT_WHITELIST: frozenset[str] = frozenset(
    {
        "thread_id",
        "status",
        "created_at",
        "updated_at",
        "metadata",
        "user_id",
        "config",
        "values",
        "interrupts",
    }
)
CHECKPOINT_SELECT_FIELDS: frozenset[str] = frozenset({"values", "interrupts", "config"})


class ThreadSearchRequest(BaseModel):
    """Request model for thread search"""

    metadata: dict[str, Any] | None = Field(None, description="Metadata filters")
    status: str | None = Field(None, description="Thread status filter (idle, busy, interrupted, error)")
    limit: int | None = Field(20, le=100, ge=1, description="Maximum results")
    offset: int | None = Field(0, ge=0, description="Results offset")
    order_by: str | None = Field(
        "created_at DESC",
        deprecated=True,
        description="DEPRECATED: use sort_by + sort_order. Legacy single-field form, e.g. 'updated_at ASC'.",
    )
    sort_by: Literal["thread_id", "status", "created_at", "updated_at"] | None = Field(
        None,
        description="Field to sort by (SDK-compatible). Takes precedence over order_by.",
    )
    sort_order: Literal["asc", "desc"] | None = Field(
        None,
        description="Sort direction (SDK-compatible). Defaults to 'desc' when sort_by is set.",
    )
    select: list[str] | None = Field(
        None,
        description=(
            "Optional response field projection. When omitted, returns the default thin "
            "thread shape. Allowed: thread_id, status, created_at, updated_at, metadata, "
            "user_id, config, values, interrupts."
        ),
    )
    extract: dict[str, str] | None = Field(
        None,
        description=(
            "Optional JSON-path projections (max 10). Keys are response aliases; values are "
            "paths starting with values., metadata., or config. "
            "Resolved values appear under 'extracted' on each result. Unknown/malformed paths → 422."
        ),
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        """Validate status filter conforms to API specification."""
        if v is not None:
            return validate_thread_status(v)
        return v

    @field_validator("select")
    @classmethod
    def validate_select(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("select must be a non-empty list when provided")
        unknown = sorted({field for field in v if field not in THREAD_SEARCH_SELECT_WHITELIST})
        if unknown:
            allowed = ", ".join(sorted(THREAD_SEARCH_SELECT_WHITELIST))
            raise ValueError(f"unknown select fields: {unknown}; allowed: {allowed}")
        # Preserve order, drop duplicates
        seen: set[str] = set()
        deduped: list[str] = []
        for field in v:
            if field not in seen:
                seen.add(field)
                deduped.append(field)
        return deduped

    @field_validator("extract")
    @classmethod
    def validate_extract(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return v
        from aegra_api.utils.json_path import validate_extract_paths

        validate_extract_paths(v)
        return v


class ThreadSearchItem(Thread):
    """Documented shape for projected ``POST /threads/search`` results.

    Used for OpenAPI/SDK typing of optional ``values`` / ``interrupts`` /
    ``config`` / ``extracted`` fields. Not used as a validating
    ``response_model`` because sparse ``select`` projections may omit
    inherited required Thread fields.
    """

    values: dict[str, Any] | None = Field(None, description="Latest checkpoint channel values")
    interrupts: list[dict[str, Any]] | None = Field(None, description="Pending interrupts from latest state")
    config: dict[str, Any] | None = Field(None, description="Latest runnable config from checkpoint")
    extracted: dict[str, Any] | None = Field(None, description="Values resolved from extract paths")


class ThreadSearchResponse(BaseModel):
    """Response model for thread search"""

    threads: list[Thread]
    total: int
    limit: int
    offset: int


class ThreadCheckpoint(BaseModel):
    """Checkpoint identifier for thread history"""

    checkpoint_id: str | None = None
    thread_id: str | None = None
    checkpoint_ns: str | None = ""


class ThreadCheckpointPostRequest(BaseModel):
    """Request model for fetching thread checkpoint"""

    checkpoint: ThreadCheckpoint = Field(description="Checkpoint to fetch")
    subgraphs: bool | None = Field(False, description="Include subgraph states")


class ThreadState(BaseModel):
    """Thread state model for history endpoint"""

    values: dict[str, Any] = Field(description="Channel values (messages, etc.)")
    next: list[str] = Field(default_factory=list, description="Next nodes to execute")
    tasks: list[dict[str, Any]] = Field(default_factory=list, description="Tasks to execute")
    interrupts: list[dict[str, Any]] = Field(default_factory=list, description="Interrupt data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Checkpoint metadata")
    created_at: datetime | None = Field(None, description="Timestamp of state creation")
    checkpoint: ThreadCheckpoint = Field(description="Current checkpoint")
    parent_checkpoint: ThreadCheckpoint | None = Field(None, description="Parent checkpoint")
    checkpoint_id: str | None = Field(None, description="Checkpoint ID (for backward compatibility)")
    parent_checkpoint_id: str | None = Field(None, description="Parent checkpoint ID (for backward compatibility)")


class ThreadStateUpdate(BaseModel):
    """Request model for updating thread state"""

    values: dict[str, Any] | list[dict[str, Any]] | None = Field(
        None, description="The values to update the state with"
    )
    checkpoint: dict[str, Any] | None = Field(None, description="The checkpoint to update the state of")
    checkpoint_id: str | None = Field(None, description="Optional checkpoint ID to update from")
    as_node: str | None = Field(None, description="Update the state as if this node had just executed")
    # Also support query-like parameters for GET-like behavior via POST
    subgraphs: bool | None = Field(False, description="Include states from subgraphs")
    checkpoint_ns: str | None = Field(None, description="Checkpoint namespace")


class ThreadStateUpdateResponse(BaseModel):
    """Response model for thread state update"""

    checkpoint: dict[str, Any] = Field(description="The checkpoint that was created/updated")


class ThreadHistoryRequest(BaseModel):
    """Request model for thread history endpoint"""

    limit: int | None = Field(10, ge=1, le=1000, description="Number of states to return")
    before: dict[str, Any] | str | None = Field(
        None,
        description="Return states before this checkpoint (checkpoint ID string, raw checkpoint dict, or RunnableConfig with 'configurable' key)",
    )
    metadata: dict[str, Any] | None = Field(None, description="Filter by metadata")
    checkpoint: dict[str, Any] | None = Field(None, description="Checkpoint for subgraph filtering")
    subgraphs: bool | None = Field(False, description="Include states from subgraphs")
    checkpoint_ns: str | None = Field(None, description="Checkpoint namespace")
