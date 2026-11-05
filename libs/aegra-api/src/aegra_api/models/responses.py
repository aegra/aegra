"""Open Responses request and response models.

This surface is not the Agent Protocol. Fields that would change OpenAI
behavior and are not implemented are rejected when set to a non-default value.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_METADATA_MAX_KEYS = 16
_METADATA_MAX_KEY_LEN = 64
_METADATA_MAX_VALUE_LEN = 512

_UNSUPPORTED_WHEN_SET = (
    "temperature",
    "top_p",
    "max_output_tokens",
    "truncation",
    "reasoning",
    "text",
    "prompt",
)


class ResponsesError(Exception):
    """Error rendered as an OpenAI ``{error: {message, type, code}}`` body."""

    def __init__(self, status_code: int, message: str, *, code: str | None = None) -> None:
        self.status_code = status_code
        self.message = message
        self.code = code or code_for_status(status_code)
        super().__init__(message)

    def body(self) -> dict[str, Any]:
        return openai_error_body(self.status_code, self.message, code=self.code)


def code_for_status(status_code: int) -> str:
    """Stable ``error.code`` for an HTTP status on this surface."""
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 403:
        return "permission_denied"
    if status_code == 401:
        return "unauthorized"
    if status_code >= 500:
        return "server_error"
    return "invalid_request"


def type_for_status(status_code: int) -> str:
    """OpenAI ``error.type`` for an HTTP status."""
    if status_code == 401 or status_code == 403:
        return "permission_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def openai_error_body(status_code: int, message: str, *, code: str | None = None) -> dict[str, Any]:
    """Build the OpenAI error document."""
    return {
        "error": {
            "message": message,
            "type": type_for_status(status_code),
            "code": code or code_for_status(status_code),
        }
    }


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object of strings")
    if len(metadata) > _METADATA_MAX_KEYS:
        raise ValueError(f"metadata exceeds {_METADATA_MAX_KEYS} keys")
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("metadata keys and values must be strings")
        if len(key) > _METADATA_MAX_KEY_LEN or len(value) > _METADATA_MAX_VALUE_LEN:
            raise ValueError("metadata key or value is too long")


class ResponseCreate(BaseModel):
    """POST /v1/responses body. Null SDK defaults are accepted."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    conversation: str | dict[str, Any] | None = None
    previous_response_id: str | None = None
    stream: bool = False
    background: bool = False
    store: bool = True
    metadata: dict[str, Any] | None = None
    user: str | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    truncation: str | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    prompt: Any | None = None

    @model_validator(mode="after")
    def reject_unsupported(self) -> Self:
        """Reject values this facade cannot honor instead of ignoring them."""
        if self.tools:
            raise ValueError("tools is not supported")
        if self.tool_choice not in (None, "auto", "none"):
            raise ValueError("tool_choice is not supported unless it is 'auto' or 'none'")
        for name in _UNSUPPORTED_WHEN_SET:
            if getattr(self, name) is not None:
                raise ValueError(f"{name} is not supported")
        if self.background and not self.store:
            raise ValueError("background=true requires store=true")
        if self.input is None:
            raise ValueError("input is required")
        if self.metadata is not None:
            _validate_metadata(self.metadata)
        for key, value in (self.model_extra or {}).items():
            if value is not None:
                raise ValueError(f"{key} is not supported")
        return self


class ConversationCreate(BaseModel):
    """POST /v1/conversations body."""

    model_config = ConfigDict(extra="allow")

    metadata: dict[str, Any] | None = None
    items: list[Any] | None = None

    @model_validator(mode="after")
    def reject_unsupported(self) -> Self:
        if self.items:
            raise ValueError("seeding conversation items is not supported")
        if self.metadata is not None:
            _validate_metadata(self.metadata)
        for key, value in (self.model_extra or {}).items():
            if value is not None:
                raise ValueError(f"{key} is not supported")
        return self


class ResponseResource(BaseModel):
    """Response object returned by /v1/responses."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: str
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None
    instructions: str | None = None
    model: str
    output: list[dict[str, Any]] = Field(default_factory=list)
    previous_response_id: str | None = None
    conversation: dict[str, str] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    background: bool = False
    user: str | None = None


class ConversationResource(BaseModel):
    """Conversation object. The id is the underlying thread id."""

    id: str
    object: Literal["conversation"] = "conversation"
    created_at: int
    metadata: dict[str, str] = Field(default_factory=dict)


class ConversationItemList(BaseModel):
    """GET /v1/conversations/{id}/items body."""

    object: Literal["list"] = "list"
    data: list[dict[str, Any]] = Field(default_factory=list)


class OpenAIErrorBody(BaseModel):
    """OpenAI error document returned by this surface."""

    message: str
    type: str
    code: str


class OpenAIErrorResponse(BaseModel):
    """Wrapper around :class:`OpenAIErrorBody`."""

    error: OpenAIErrorBody


class ConversationDeleted(BaseModel):
    """DELETE /v1/conversations/{id} body."""

    id: str
    object: Literal["conversation.deleted"] = "conversation.deleted"
    deleted: bool = True
