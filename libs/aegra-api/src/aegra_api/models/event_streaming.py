"""Request models for Agent Protocol v2 event streaming endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EventStreamRequest(BaseModel):
    """Body for ``POST /threads/{thread_id}/stream/events``."""

    run_id: str = Field(..., description="Run on the thread whose events to stream.")
    channels: list[str] = Field(..., description="Channels to subscribe to (e.g. messages, values, lifecycle).")
    since: int | None = Field(
        None,
        ge=0,
        description="Last seq the client saw; events at or below it are skipped on resume.",
    )


class ThreadCommand(BaseModel):
    """Body for ``POST /threads/{thread_id}/commands`` (JSON-RPC style)."""

    id: int = Field(..., description="Client-assigned command id, echoed in the response.")
    method: str = Field(..., description="Command method, e.g. run.start or input.respond.")
    params: dict[str, Any] = Field(default_factory=dict, description="Method parameters.")
