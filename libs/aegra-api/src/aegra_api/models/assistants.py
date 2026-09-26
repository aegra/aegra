"""Assistant-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aegra_api.models.entity_ids import ENTITY_ID_PATTERN, MAX_ENTITY_ID_LENGTH


class AssistantCreate(BaseModel):
    """Request model for creating assistants"""

    assistant_id: str | None = Field(
        None,
        min_length=1,
        max_length=MAX_ENTITY_ID_LENGTH,
        pattern=ENTITY_ID_PATTERN,
        description=(
            "Optional client-provided assistant ID. "
            "Omit or null to let the server generate a UUID. "
            f"When set, 1-{MAX_ENTITY_ID_LENGTH} characters and not blank "
            "(must fit PostgreSQL btree keys uncompressed)."
        ),
    )
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


def _publish_the_not_null_columns(schema: dict[str, Any]) -> None:
    """Publish ``name`` and ``graph_id`` as the endpoint takes them: never null.

    Both back NOT NULL columns and are answered 422 when supplied empty. They
    stay nullable on the model so a null arrives as supplied and gets that
    answer rather than a generic type error.
    """
    for field in ("name", "graph_id"):
        published = schema["properties"][field]
        published.pop("anyOf", None)
        published.pop("default", None)
        published["type"] = "string"


class AssistantUpdate(BaseModel):
    """Request model for partially updating assistants.

    Every field is optional and defaults to ``None`` so that an omitted field
    is distinguishable from an explicit one via ``model_dump(exclude_unset=True)``:
    omitting ``config`` keeps the stored config, sending ``{"config": {}}`` clears it.

    A null clears ``description`` and reads as empty for the other dict fields.
    ``name`` and ``graph_id`` back NOT NULL columns and refuse one.
    """

    name: str | None = Field(None, description="The name of the assistant. Unchanged when omitted.")
    description: str | None = Field(None, description="The description of the assistant. Unchanged when omitted.")
    config: dict[str, Any] | None = Field(
        None, description="Configuration to use for the graph. Unchanged when omitted."
    )
    graph_id: str | None = Field(None, description="The ID of the graph. Unchanged when omitted.")
    context: dict[str, Any] | None = Field(
        None,
        description="The context to use for the graph. Useful when graph is configurable. Unchanged when omitted.",
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Metadata to merge into the assistant's existing metadata."
    )

    model_config = ConfigDict(json_schema_extra=_publish_the_not_null_columns)


class AssistantList(BaseModel):
    """Response model for listing assistants"""

    assistants: list[Assistant]
    total: int


class AssistantSearchRequest(BaseModel):
    """Request model for assistant search"""

    name: str | None = Field(None, description="Filter by assistant name")
    description: str | None = Field(None, description="Filter by assistant description")
    graph_id: str | None = Field(None, description="Filter by graph ID")
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
