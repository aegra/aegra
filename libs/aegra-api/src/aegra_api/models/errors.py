"""Error response models for Agent Protocol"""

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field


class AgentProtocolError(BaseModel):
    """Standard Agent Protocol error response"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")


class DetailedHTTPException(HTTPException):
    """An ``HTTPException`` that fills in ``AgentProtocolError.details``.

    ``HTTPException`` declares no such attribute, so raising this is what lets
    ``agent_protocol_exception_handler`` find one.
    """

    def __init__(self, status_code: int, detail: str, details: dict[str, Any]) -> None:
        super().__init__(status_code, detail=detail)
        self.details = details


# Reusable OpenAPI error response declarations for endpoint decorators
NOT_FOUND: dict[int, dict[str, object]] = {
    404: {"model": AgentProtocolError, "description": "Resource not found"},
}
CONFLICT: dict[int, dict[str, object]] = {
    409: {"model": AgentProtocolError, "description": "Resource state conflict"},
}
BAD_REQUEST: dict[int, dict[str, object]] = {
    400: {"model": AgentProtocolError, "description": "Invalid request"},
}
UNAVAILABLE: dict[int, dict[str, object]] = {
    503: {"model": AgentProtocolError, "description": "Service unavailable"},
}

# SSE streaming response for OpenAPI documentation
SSE_RESPONSE: dict[int, dict[str, object]] = {
    200: {
        "description": "Server-Sent Events stream",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    },
}


def get_error_type(status_code: int) -> str:
    """Map HTTP status codes to error types"""
    error_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        500: "internal_error",
        501: "not_implemented",
        503: "service_unavailable",
    }
    return error_map.get(status_code, "unknown_error")
