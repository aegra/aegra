"""Assistant-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssistantCreate(BaseModel):
    """Request model for creating assistants"""

    assistant_id: str | None = Field(None, description="Unique assistant identifier (auto-generated if not provided)")
    name: str | None = Field(
        None,
        description="Human-readable assistant name (auto-generated if not provided)",
    )
    description: str | None = Field(None, description="Assistant description")
    config: dict[str, Any] | None = Field(default_factory=dict, description="Assistant configuration")
    context: dict[str, Any] | None = Field(default_factory=dict, description="Assistant context")
    graph_id: str = Field(..., description="LangGraph graph ID from aegra.json")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Metadata to use for searching and filtering assistants."
    )
    if_exists: str | None = Field("error", description="What to do if assistant exists: error or do_nothing")
    secrets: dict[str, str] | None = Field(
        None, description="Named secrets (e.g. api_key) stored encrypted at rest; never returned in responses."
    )


class Assistant(BaseModel):
    """Assistant entity model"""

    assistant_id: str = Field(..., description="Unique identifier for the assistant.")
    name: str = Field(..., description="Human-readable name of the assistant.")
    description: str | None = Field(None, description="Optional description of the assistant's purpose.")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration passed to the graph at runtime.")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Context variables available to the graph during execution."
    )
    graph_id: str = Field(..., description="Identifier of the graph this assistant executes.")
    user_id: str = Field(..., description="Identifier of the user who owns this assistant.")
    version: int = Field(..., description="The version of the assistant.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, alias="metadata_dict", description="Arbitrary metadata for searching and filtering."
    )
    created_at: datetime = Field(..., description="Timestamp when the assistant was created.")
    updated_at: datetime = Field(..., description="Timestamp when the assistant was last updated.")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class AssistantUpdate(BaseModel):
    """Request model for creating assistants"""

    name: str | None = Field(None, description="The name of the assistant (auto-generated if not provided)")
    description: str | None = Field(None, description="The description of the assistant. Defaults to null.")
    config: dict[str, Any] | None = Field(default_factory=dict, description="Configuration to use for the graph.")
    graph_id: str = Field("agent", description="The ID of the graph")
    context: dict[str, Any] | None = Field(
        default_factory=dict,
        description="The context to use for the graph. Useful when graph is configurable.",
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Metadata to use for searching and filtering assistants."
    )
    secrets: dict[str, str] | None = Field(
        None, description="Named secrets (e.g. api_key) stored encrypted at rest; never returned in responses."
    )


class AssistantList(BaseModel):
    """Response model for listing assistants"""

    assistants: list[Assistant]
    total: int


class AssistantSearchRequest(BaseModel):
    """Request model for assistant search"""

    name: str | None = Field(None, description="Filter by assistant name")
    description: str | None = Field(None, description="Filter by assistant description")
    graph_id: str | None = Field(None, description="Filter by graph ID")
    user_id: str | None = Field(None, description="Filter by user_id (within the caller's authorization scope)")
    tenant_id: str | None = Field(None, description="Filter by tenant_id (within the caller's authorization scope)")
    limit: int | None = Field(20, le=100, ge=1, description="Maximum results")
    offset: int | None = Field(0, ge=0, description="Results offset")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Metadata to use for searching and filtering assistants.",
    )
    sort_by: Literal["assistant_id", "name", "graph_id", "created_at", "updated_at"] | None = Field(
        None,
        description="Field to sort by (SDK-compatible).",
    )
    sort_order: Literal["asc", "desc"] | None = Field(
        None,
        description="Sort direction (SDK-compatible). Defaults to 'desc' when sort_by is set.",
    )


class AgentSchemas(BaseModel):
    """Agent schema definitions for client integration"""

    input_schema: dict[str, Any] = Field(..., description="JSON Schema for agent inputs")
    output_schema: dict[str, Any] = Field(..., description="JSON Schema for agent outputs")
    state_schema: dict[str, Any] = Field(..., description="JSON Schema for agent state")
    config_schema: dict[str, Any] = Field(..., description="JSON Schema for agent config")


class AssistantShareCreate(BaseModel):
    """Create-share request: share_type determines which target field is required."""

    share_type: Literal["user", "tenant", "public"] = Field(..., description="Share scope: user/tenant/public")
    target_user_id: str | None = Field(None, description="Target user when share_type=user")
    target_tenant_id: str | None = Field(None, description="Target tenant when share_type=tenant")

    @model_validator(mode="after")
    def _check_target(self) -> "AssistantShareCreate":
        if self.share_type == "user" and not self.target_user_id:
            raise ValueError("share_type=user requires target_user_id")
        if self.share_type == "tenant" and not self.target_tenant_id:
            raise ValueError("share_type=tenant requires target_tenant_id")
        if self.share_type == "public" and (self.target_user_id or self.target_tenant_id):
            raise ValueError("share_type=public must not carry a target")
        return self


class AssistantShareResponse(BaseModel):
    """Share record response."""

    model_config = ConfigDict(from_attributes=True)

    share_id: str
    assistant_id: str
    share_type: str
    target_user_id: str | None = None
    target_tenant_id: str | None = None
    created_at: datetime
