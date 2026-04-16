"""Pydantic models for cron job endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class CronCreate(BaseModel):
    """Request body for creating a cron job (stateless or thread-bound)."""

    assistant_id: str
    schedule: str
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    checkpoint_during: bool | None = None
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    webhook: str | None = None
    on_run_completed: str | None = None
    multitask_strategy: str | None = None
    end_time: datetime | None = None
    enabled: bool | None = None
    stream_mode: str | list[str] | None = None
    stream_subgraphs: bool | None = None
    stream_resumable: bool | None = None
    durability: str | None = None
    timezone: str | None = None


class CronResponse(BaseModel):
    """Response model matching the SDK ``Cron`` TypedDict."""

    model_config = ConfigDict(from_attributes=True)

    cron_id: str
    assistant_id: str
    thread_id: str | None = None
    on_run_completed: str | None = None
    end_time: datetime | None = None
    schedule: str
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any] = {}
    user_id: str | None = None
    next_run_date: datetime | None = None
    metadata: dict[str, Any] = {}
    enabled: bool = True


class CronUpdate(BaseModel):
    """Request body for updating an existing cron job."""

    schedule: str | None = None
    end_time: datetime | None = None
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    checkpoint_during: bool | None = None
    webhook: str | None = None
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    on_run_completed: str | None = None
    multitask_strategy: str | None = None
    enabled: bool | None = None
    stream_mode: str | list[str] | None = None
    stream_subgraphs: bool | None = None
    stream_resumable: bool | None = None
    durability: str | None = None
    timezone: str | None = None


class CronSearchRequest(BaseModel):
    """Request body for searching cron jobs."""

    assistant_id: str | None = None
    thread_id: str | None = None
    enabled: bool | None = None
    limit: int = 10
    offset: int = 0
    sort_by: str | None = None
    sort_order: str | None = None
    select: list[str] | None = None


class CronCountRequest(BaseModel):
    """Request body for counting cron jobs."""

    assistant_id: str | None = None
    thread_id: str | None = None
