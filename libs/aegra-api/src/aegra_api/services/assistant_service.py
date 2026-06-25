"""Service layer for assistant business logic

This service encapsulates all business logic for assistant management, following
a layered architecture pattern. The code was extracted from api/assistants.py
to separate concerns and improve maintainability.

Responsibilities:
- Business logic and validation
- Database operations via SQLAlchemy ORM
- Graph schema extraction and manipulation
- Coordination between different components

This is the first service layer implementation in Aegra. The pattern will be
applied to other APIs (runs, threads, crons) as part of ongoing refactoring.
"""

import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException
from langchain_core.runnables.utils import create_model
from pydantic import TypeAdapter
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_deps import get_current_user
from aegra_api.core.auth_filters import build_metadata_filter
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import AssistantVersion as AssistantVersionORM
from aegra_api.core.orm import get_session
from aegra_api.models import Assistant, AssistantCreate, AssistantUpdate
from aegra_api.models.auth import User
from aegra_api.services.authenticated import Authenticated
from aegra_api.services.langgraph_service import LangGraphService, get_langgraph_service


def to_pydantic(row: AssistantORM) -> Assistant:
    """Convert SQLAlchemy ORM object to Pydantic model with proper type casting.

    Uses from_attributes=True because Assistant ORM has attribute/column name mismatch:
    - ORM attribute: metadata_dict
    - DB column: metadata
    - Pydantic field: metadata (with alias="metadata_dict")

    This is different from Thread/Run where attribute names match column names.
    """
    # Cast UUIDs to str so they match the Pydantic schema
    if hasattr(row, "assistant_id") and row.assistant_id is not None:
        row.assistant_id = str(row.assistant_id)
    if hasattr(row, "user_id") and isinstance(row.user_id, uuid.UUID):
        row.user_id = str(row.user_id)

    # Use Pydantic's built-in ORM conversion with from_attributes=True
    return Assistant.model_validate(row, from_attributes=True)


def _state_jsonschema(graph) -> dict | None:
    """Extract state schema from graph channels"""
    fields: dict = {}
    for k in graph.stream_channels_list:
        v = graph.channels[k]
        try:
            create_model(k, __root__=(v.UpdateType, None)).model_json_schema()
            fields[k] = (v.UpdateType, None)
        except Exception:
            fields[k] = (Any, None)
    return create_model(graph.get_name("State"), **fields).model_json_schema()


def _get_configurable_jsonschema(graph) -> dict:
    """Get the JSON schema for the configurable part of the graph"""
    EXCLUDED_CONFIG_SCHEMA = {"__pregel_resuming", "__pregel_checkpoint_id"}

    config_schema = graph.config_schema()
    model_fields = getattr(config_schema, "model_fields", None) or getattr(config_schema, "__fields__", None)

    if model_fields is not None and "configurable" in model_fields:
        configurable = TypeAdapter(model_fields["configurable"].annotation)
        json_schema = configurable.json_schema()
        if json_schema:
            for key in EXCLUDED_CONFIG_SCHEMA:
                json_schema["properties"].pop(key, None)
        if hasattr(graph, "config_type") and graph.config_type is not None and hasattr(graph.config_type, "__name__"):
            json_schema["title"] = graph.config_type.__name__
        return json_schema
    return {}


def _extract_graph_schemas(graph) -> dict:
    """Extract schemas from a compiled LangGraph graph object"""
    try:
        input_schema = graph.get_input_jsonschema()
    except Exception:
        input_schema = None

    try:
        output_schema = graph.get_output_jsonschema()
    except Exception:
        output_schema = None

    try:
        state_schema = _state_jsonschema(graph)
    except Exception:
        state_schema = None

    try:
        config_schema = _get_configurable_jsonschema(graph)
    except Exception:
        config_schema = None

    try:
        context_schema = graph.get_context_jsonschema()
    except Exception:
        context_schema = None

    return {
        "input_schema": input_schema,
        "output_schema": output_schema,
        "state_schema": state_schema,
        "config_schema": config_schema,
        "context_schema": context_schema,
    }


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``, ``\\``) in user input.
    Backslash is replaced first so subsequent escapes are not double-escaped.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _injected_metadata(
    base: dict[str, Any] | None,
    value: dict[str, Any],
) -> dict[str, Any] | None:
    """Fold metadata a create/update handler injected into ``value`` onto the request.

    Handlers inject by mutating ``value["metadata"]`` in place (e.g.
    ``value["metadata"]["created_by"] = ctx.user.identity``). The handler's
    *return* is a query filter with no insert meaning, so it is not read here.
    """
    value_meta = value.get("metadata")
    if isinstance(value_meta, dict) and value_meta:
        return {**(base or {}), **value_meta}
    return base


class AssistantService(Authenticated):
    """Service for managing assistants"""

    resource = "assistants"

    def __init__(self, session: AsyncSession, user: User, langgraph_service: LangGraphService):
        super().__init__(session, user)
        self.langgraph_service = langgraph_service

    async def create_assistant(self, request: AssistantCreate) -> Assistant:
        """Create a new assistant"""
        value = request.model_dump()
        await self._dispatch("create", value)
        request.metadata = _injected_metadata(request.metadata, value)

        available_graphs = self.langgraph_service.list_graphs()

        # Use graph_id as the main identifier
        graph_id = request.graph_id

        if graph_id not in available_graphs:
            raise HTTPException(
                400,
                f"Graph '{graph_id}' not found in aegra.json. Available: {list(available_graphs.keys())}",
            )

        # Validate graph can be loaded
        try:
            await self.langgraph_service.get_graph_for_validation(graph_id)
        except Exception as e:
            raise HTTPException(400, f"Failed to load graph: {str(e)}") from e

        config = request.config
        context = request.context

        if config.get("configurable") and context:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both configurable and context. Prefer setting context alone. Context was introduced in LangGraph 0.6.0 and is the long term planned replacement for configurable.",
            )

        # Keep config and context up to date with one another
        if config.get("configurable"):
            context = config["configurable"]
        elif context:
            config["configurable"] = context

        # Generate assistant_id if not provided
        assistant_id = request.assistant_id or str(uuid4())

        # Generate name if not provided
        name = request.name or f"Assistant for {graph_id}"

        # Check if an assistant already exists for this user, graph and config pair
        existing_stmt = select(AssistantORM).where(
            AssistantORM.user_id == self.user.identity,
            or_(
                (AssistantORM.graph_id == graph_id) & (AssistantORM.config == config),
                AssistantORM.assistant_id == assistant_id,
            ),
        )
        existing = await self.session.scalar(existing_stmt)

        if existing:
            if request.if_exists == "do_nothing":
                return to_pydantic(existing)
            else:  # error (default)
                raise HTTPException(409, f"Assistant '{assistant_id}' already exists")

        # Create assistant record
        assistant_orm = AssistantORM(
            assistant_id=assistant_id,
            name=name,
            description=request.description,
            config=config,
            context=context,
            graph_id=graph_id,
            user_id=self.user.identity,
            metadata_dict=request.metadata,
            version=1,
        )

        self.session.add(assistant_orm)
        await self.session.commit()
        await self.session.refresh(assistant_orm)

        # Create initial version record
        assistant_version_orm = AssistantVersionORM(
            assistant_id=assistant_id,
            version=1,
            graph_id=graph_id,
            config=config,
            context=context,
            created_at=datetime.now(UTC),
            name=name,
            description=request.description,
            metadata_dict=request.metadata,
        )
        self.session.add(assistant_version_orm)
        await self.session.commit()

        return to_pydantic(assistant_orm)

    async def list_assistants(self) -> list[Assistant]:
        """List user's assistants and system assistants.

        Listing dispatches the ``search`` action. A handler may scope results
        via a metadata containment filter. Unlike ``search_assistants``, this
        method does not paginate.
        """
        value: dict[str, Any] = {}
        filters = await self._dispatch("search", value)

        stmt = select(AssistantORM).where(
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system")
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        result = await self.session.scalars(stmt)
        return [to_pydantic(a) for a in result.all()]

    async def search_assistants(
        self,
        request: Any,  # AssistantSearchRequest
        *,
        sort_column: Any | None = None,
        sort_asc: bool = False,
    ) -> list[Assistant]:
        """Search assistants with filters"""
        value = request.model_dump()
        filters = await self._dispatch("search", value)

        stmt = select(AssistantORM).where(
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system")
        )

        if request.name:
            stmt = stmt.where(AssistantORM.name.ilike(f"%{_escape_like(request.name)}%", escape="\\"))

        if request.description:
            stmt = stmt.where(AssistantORM.description.ilike(f"%{_escape_like(request.description)}%", escape="\\"))

        if request.graph_id:
            stmt = stmt.where(AssistantORM.graph_id == request.graph_id)

        if request.metadata:
            stmt = stmt.where(AssistantORM.metadata_dict.op("@>")(request.metadata))

        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)

        column = sort_column if sort_column is not None else AssistantORM.created_at
        direction = column.asc() if sort_asc else column.desc()
        # Tie-break on assistant_id keeps offset pagination stable when the
        # primary sort column has duplicates.
        stmt = stmt.order_by(direction, AssistantORM.assistant_id.asc())

        offset = request.offset or 0
        limit = request.limit or 20
        stmt = stmt.offset(offset).limit(limit)

        result = await self.session.scalars(stmt)
        return [to_pydantic(a) for a in result.all()]

    async def count_assistants(self, request: Any) -> int:
        """Count assistants with filters"""
        value = request.model_dump()
        filters = await self._dispatch("search", value)

        # Include both user's assistants and system assistants (like search_assistants does)
        stmt = select(func.count()).where(
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system")
        )

        if request.name:
            stmt = stmt.where(AssistantORM.name.ilike(f"%{_escape_like(request.name)}%", escape="\\"))

        if request.description:
            stmt = stmt.where(AssistantORM.description.ilike(f"%{_escape_like(request.description)}%", escape="\\"))

        if request.graph_id:
            stmt = stmt.where(AssistantORM.graph_id == request.graph_id)

        if request.metadata:
            stmt = stmt.where(AssistantORM.metadata_dict.op("@>")(request.metadata))

        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)

        total = await self.session.scalar(stmt)
        return total or 0

    async def _read_owned_assistant(self, assistant_id: str) -> AssistantORM:
        """Dispatch ``assistants.read`` and load the row, applying any handler
        filter to the query. 404s if the row is absent or the filter excludes it.

        Shared by every read-derived endpoint so a handler's metadata filter is
        enforced uniformly — not only on GET /assistants/{id}.
        """
        filters = await self._dispatch("read", {"assistant_id": assistant_id})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system"),
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)

        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")
        return assistant

    async def get_assistant(self, assistant_id: str) -> Assistant:
        """Get assistant by ID"""
        return to_pydantic(await self._read_owned_assistant(assistant_id))

    async def update_assistant(self, assistant_id: str, request: AssistantUpdate) -> Assistant:
        """Update assistant by ID"""
        value = {**request.model_dump(), "assistant_id": assistant_id}
        filters = await self._dispatch("update", value)
        request.metadata = _injected_metadata(request.metadata, value)

        metadata = request.metadata or {}
        config = request.config or {}
        context = request.context or {}

        if config.get("configurable") and context:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both configurable and context. Use only one.",
            )

        # Keep config and context up to date with one another
        if config.get("configurable"):
            context = config["configurable"]
        elif context:
            config["configurable"] = context

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            AssistantORM.user_id == self.user.identity,
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        now = datetime.now(UTC)
        version_stmt = select(func.max(AssistantVersionORM.version)).where(
            AssistantVersionORM.assistant_id == assistant_id
        )
        max_version = await self.session.scalar(version_stmt)
        new_version = (max_version or 1) + 1 if max_version is not None else 1

        new_version_details = {
            "assistant_id": assistant_id,
            "version": new_version,
            "graph_id": request.graph_id or assistant.graph_id,
            "config": config,
            "context": context,
            "created_at": now,
            "name": request.name or assistant.name,
            "description": request.description or assistant.description,
            "metadata_dict": metadata,
        }

        assistant_version_orm = AssistantVersionORM(**new_version_details)
        self.session.add(assistant_version_orm)
        await self.session.commit()

        assistant_update = (
            update(AssistantORM)
            .where(
                AssistantORM.assistant_id == assistant_id,
                AssistantORM.user_id == self.user.identity,
            )
            .values(
                name=new_version_details["name"],
                description=new_version_details["description"],
                graph_id=new_version_details["graph_id"],
                config=new_version_details["config"],
                context=new_version_details["context"],
                metadata_dict=new_version_details["metadata_dict"],
                version=new_version,
                updated_at=now,
            )
        )
        await self.session.execute(assistant_update)
        await self.session.commit()
        updated_assistant = await self.session.scalar(stmt)
        return to_pydantic(updated_assistant)

    async def delete_assistant(self, assistant_id: str) -> dict:
        """Delete assistant by ID"""
        filters = await self._dispatch("delete", {"assistant_id": assistant_id})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            AssistantORM.user_id == self.user.identity,
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)

        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        await self.session.delete(assistant)
        await self.session.commit()

        return {"status": "deleted"}

    async def set_assistant_latest(self, assistant_id: str, version: int) -> Assistant:
        """Set the given version as the latest version of an assistant"""
        filters = await self._dispatch("update", {"assistant_id": assistant_id, "version": version})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            AssistantORM.user_id == self.user.identity,
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        version_stmt = select(AssistantVersionORM).where(
            AssistantVersionORM.assistant_id == assistant_id,
            AssistantVersionORM.version == version,
        )
        assistant_version = await self.session.scalar(version_stmt)
        if not assistant_version:
            raise HTTPException(404, f"Version '{version}' for Assistant '{assistant_id}' not found")

        assistant_update = (
            update(AssistantORM)
            .where(
                AssistantORM.assistant_id == assistant_id,
                AssistantORM.user_id == self.user.identity,
            )
            .values(
                name=assistant_version.name,
                description=assistant_version.description,
                config=assistant_version.config,
                context=assistant_version.context,
                graph_id=assistant_version.graph_id,
                metadata_dict=assistant_version.metadata_dict,
                version=version,
                updated_at=datetime.now(UTC),
            )
        )
        await self.session.execute(assistant_update)
        await self.session.commit()
        updated_assistant = await self.session.scalar(stmt)
        return to_pydantic(updated_assistant)

    async def list_assistant_versions(self, assistant_id: str) -> list[Assistant]:
        """List all versions of an assistant"""
        # Versions dispatches `search` (not `read`) per the auth dispatch spec,
        # with the {assistant_id, metadata} value shape.
        filters = await self._dispatch("search", {"assistant_id": assistant_id, "metadata": None})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system"),
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        stmt = (
            select(AssistantVersionORM)
            .where(AssistantVersionORM.assistant_id == assistant_id)
            .order_by(AssistantVersionORM.version.desc())
        )
        result = await self.session.scalars(stmt)
        versions = result.all()

        if not versions:
            raise HTTPException(404, f"No versions found for Assistant '{assistant_id}'")

        # Convert to Pydantic models
        version_list = [
            Assistant(
                assistant_id=assistant_id,
                name=v.name,
                description=v.description,
                config=v.config or {},
                context=v.context or {},
                graph_id=v.graph_id,
                user_id=self.user.identity,
                version=v.version,
                created_at=v.created_at,
                updated_at=v.created_at,
                metadata_dict=v.metadata_dict or {},
            )
            for v in versions
        ]

        return version_list

    async def get_assistant_schemas(self, assistant_id: str) -> dict[str, Any]:
        """Get input, output, state, config and context schemas for an assistant"""
        assistant = await self._read_owned_assistant(assistant_id)

        try:
            # Use get_graph_for_validation since we only need schema extraction,
            # not checkpointer/store for execution
            graph = await self.langgraph_service.get_graph_for_validation(
                assistant.graph_id,
                user=self.user,
            )
            schemas = _extract_graph_schemas(graph)

            return {"graph_id": assistant.graph_id, **schemas}

        except Exception as e:
            raise HTTPException(400, f"Failed to extract schemas: {str(e)}") from e

    async def get_assistant_graph(self, assistant_id: str, xray: bool | int) -> dict[str, Any]:
        """Get the graph structure for visualization"""
        assistant = await self._read_owned_assistant(assistant_id)

        try:
            # Use get_graph_for_validation since we only need graph structure,
            # not checkpointer/store for execution
            graph = await self.langgraph_service.get_graph_for_validation(
                assistant.graph_id,
                user=self.user,
            )

            # Validate xray if it's an integer (not a boolean)
            if isinstance(xray, int) and not isinstance(xray, bool) and xray <= 0:
                raise HTTPException(422, detail="Invalid xray value")

            try:
                drawable_graph = await graph.aget_graph(xray=xray)
                json_graph = drawable_graph.to_json()

                for node in json_graph.get("nodes", []):
                    if (data := node.get("data")) and isinstance(data, dict):
                        data.pop("id", None)

                return json_graph
            except NotImplementedError as e:
                raise HTTPException(422, detail="The graph does not support visualization") from e

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to get graph: {str(e)}") from e

    async def get_assistant_subgraphs(
        self,
        assistant_id: str,
        namespace: str | None,
        recurse: bool,
    ) -> dict[str, Any]:
        """Get subgraphs of an assistant"""
        assistant = await self._read_owned_assistant(assistant_id)

        try:
            # Use get_graph_for_validation since we only need schema extraction,
            # not checkpointer/store for execution
            graph = await self.langgraph_service.get_graph_for_validation(
                assistant.graph_id,
                user=self.user,
            )

            try:
                subgraphs = {
                    ns: _extract_graph_schemas(subgraph)
                    async for ns, subgraph in graph.aget_subgraphs(namespace=namespace, recurse=recurse)
                }
                return subgraphs
            except NotImplementedError as e:
                raise HTTPException(422, detail="The graph does not support subgraphs") from e

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to get subgraphs: {str(e)}") from e


def get_assistant_service(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
) -> AssistantService:
    """Dependency injection for AssistantService"""
    return AssistantService(session, user, langgraph_service)
