"""Pydantic models for cron job endpoints."""

from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegra_api.settings import settings

# Field length caps. Keep these conservative; cron metadata is small by nature.
_SCHEDULE_MAX_LEN = 256
_TIMEZONE_MAX_LEN = 64
_WEBHOOK_MAX_LEN = 2048
_STREAM_MODE_MAX_LEN = 64
_STR_FIELD_MAX_LEN = 256

OnRunCompleted = Literal["delete", "keep"]


def _validate_webhook_url(value: str | None) -> str | None:
    """Reject malformed or non-http(s) webhook URLs at the API boundary.

    The cron record stores ``webhook`` verbatim. When a future runtime wires
    webhook delivery this is the SSRF entry point, so we constrain it now.
    """
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("webhook must use http or https scheme")
    if not parsed.netloc:
        raise ValueError("webhook must include a host")
    return value


def _validate_payload_size(model: BaseModel) -> None:
    """Reject payloads whose serialized JSON exceeds the configured cap."""
    cap = settings.cron.CRON_MAX_PAYLOAD_BYTES
    serialized = model.model_dump_json()
    if len(serialized.encode("utf-8")) > cap:
        raise ValueError(f"cron payload exceeds {cap} bytes")


class CronCreate(BaseModel):
    """Request body for creating a cron job (stateless or thread-bound)."""

    assistant_id: str = Field(..., max_length=_STR_FIELD_MAX_LEN)
    schedule: str = Field(..., max_length=_SCHEDULE_MAX_LEN)
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    checkpoint_during: bool | None = None
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    webhook: str | None = Field(None, max_length=_WEBHOOK_MAX_LEN)
    on_run_completed: OnRunCompleted | None = None
    multitask_strategy: str | None = Field(None, max_length=_STR_FIELD_MAX_LEN)
    end_time: datetime | None = None
    enabled: bool | None = None
    stream_mode: str | list[str] | None = None
    stream_subgraphs: bool | None = None
    stream_resumable: bool | None = None
    durability: str | None = Field(None, max_length=_STR_FIELD_MAX_LEN)
    timezone: str | None = Field(None, max_length=_TIMEZONE_MAX_LEN)

    @model_validator(mode="after")
    def _check(self) -> "CronCreate":
        self.webhook = _validate_webhook_url(self.webhook)
        if isinstance(self.stream_mode, str) and len(self.stream_mode) > _STREAM_MODE_MAX_LEN:
            raise ValueError("stream_mode is too long")
        if self.end_time is not None:
            now = datetime.now(UTC)
            end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=UTC)
            if end <= now:
                raise ValueError("end_time must be in the future")
        _validate_payload_size(self)
        return self


class CronResponse(BaseModel):
    """Response model matching the SDK ``Cron`` TypedDict."""

    model_config = ConfigDict(from_attributes=True)

    cron_id: str
    assistant_id: str
    thread_id: str | None = None
    on_run_completed: OnRunCompleted | None = None
    end_time: datetime | None = None
    schedule: str
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    next_run_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class CronUpdate(BaseModel):
    """Request body for updating an existing cron job."""

    schedule: str | None = Field(None, max_length=_SCHEDULE_MAX_LEN)
    end_time: datetime | None = None
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    checkpoint_during: bool | None = None
    webhook: str | None = Field(None, max_length=_WEBHOOK_MAX_LEN)
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    on_run_completed: OnRunCompleted | None = None
    multitask_strategy: str | None = Field(None, max_length=_STR_FIELD_MAX_LEN)
    enabled: bool | None = None
    stream_mode: str | list[str] | None = None
    stream_subgraphs: bool | None = None
    stream_resumable: bool | None = None
    durability: str | None = Field(None, max_length=_STR_FIELD_MAX_LEN)
    timezone: str | None = Field(None, max_length=_TIMEZONE_MAX_LEN)

    @model_validator(mode="after")
    def _check(self) -> "CronUpdate":
        self.webhook = _validate_webhook_url(self.webhook)
        if isinstance(self.stream_mode, str) and len(self.stream_mode) > _STREAM_MODE_MAX_LEN:
            raise ValueError("stream_mode is too long")
        if self.end_time is not None:
            now = datetime.now(UTC)
            end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=UTC)
            if end <= now:
                raise ValueError("end_time must be in the future")
        _validate_payload_size(self)
        return self


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
