"""Thread-related Pydantic models for Agent Protocol"""

from base64 import b64encode
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from math import isfinite
from pathlib import Path
from re import Pattern
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

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

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        """Validate status filter conforms to API specification."""
        if v is not None:
            return validate_thread_status(v)
        return v


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


def _json_key(key: Any) -> str:
    """Render an arbitrary dict key as a JSON object key string.

    The fallback encoder never fires for dict keys, so keys that are not
    already strings must be converted explicitly. bytes/bytearray encode
    as standard Base64; other types mirror OPT_NON_STR_KEYS output.
    """
    if isinstance(key, str):
        return key
    if isinstance(key, (bytes, bytearray)):
        return b64encode(key).decode("ascii")
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, Enum):
        return str(key.value)
    if isinstance(key, (datetime, date, time)):
        return key.isoformat()
    if isinstance(key, UUID):
        return str(key)
    if key is None:
        return "null"
    return str(key)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert arbitrary thread state to a JSON-compatible value.

    Handles bytes/bytearray as standard Base64, dict keys via _json_key,
    Pydantic models via model_dump(), v1/LangChain objects via dict(),
    NamedTuples via _asdict(), dataclasses via asdict(), set/frozenset/deque
    as arrays, and other supported types. Unknown objects become null so the
    endpoints never 500 on arbitrary checkpointed state.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, (bytes, bytearray)):
        return b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {_json_key(k): _to_jsonable(v) for k, v in value.items()}
    # NamedTuple is a tuple subclass; convert via _asdict before the generic
    # container branch, otherwise it would serialize as an array.
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return _to_jsonable(value._asdict())
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _to_jsonable(value.dict())
    if isinstance(value, BaseException):
        return {"error": type(value).__name__, "message": str(value)}
    if isinstance(value, (timezone, ZoneInfo)):
        return value.tzname(None)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return int(value) if value.as_tuple().exponent >= 0 else float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, (IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network, Path)):
        return str(value)
    if isinstance(value, Pattern):
        return value.pattern
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    return None


class ThreadState(BaseModel):
    """Thread state model for history endpoint

    Binary values (``bytes``/``bytearray``) and other non-JSON-native types
    nested in arbitrary fields are encoded during JSON serialization. Dict
    keys that are bytes also encode as Base64. Python-mode access retains
    raw values.
    """

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

    @field_serializer("values", "metadata", when_used="json")
    @classmethod
    def _serialize_mapping_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _to_jsonable(value)

    @field_serializer("tasks", "interrupts", when_used="json")
    @classmethod
    def _serialize_sequence_fields(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _to_jsonable(value)


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
