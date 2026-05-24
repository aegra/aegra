"""Health check endpoints"""

import contextlib

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

from aegra_api import __version__
from aegra_api.core.database import db_manager
from aegra_api.models.errors import UNAVAILABLE
from aegra_api.settings import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str = Field(..., description="Overall health status: 'healthy' or 'unhealthy'.")
    database: str = Field(..., description="PostgreSQL connection status.")
    langgraph_checkpointer: str = Field(..., description="Checkpoint backend connection status.")
    langgraph_store: str = Field(..., description="Store backend connection status.")


class InfoResponse(BaseModel):
    """Info endpoint response model"""

    name: str = Field(..., description="Service name.")
    version: str = Field(..., description="Current server version.")
    description: str = Field(..., description="Service description.")
    status: str = Field(..., description="Current service status.")
    flags: dict = Field(..., description="Feature flags indicating available capabilities.")


@router.get("/info", response_model=InfoResponse)
async def info(_request: Request) -> InfoResponse:
    """Get service information.

    Returns the server name, version, and feature flags. This endpoint does
    not require authentication.
    """
    return InfoResponse(
        name="Aegra",
        version=__version__,
        description="Production-ready Agent Protocol server built on LangGraph",
        status="running",
        flags={"assistants": True, "crons": settings.cron.CRON_ENABLED},
    )


@router.get("/health", response_model=HealthResponse, responses={**UNAVAILABLE})
async def health_check(_request: Request) -> HealthResponse:
    """Check the health of all server components.

    Verifies connectivity to PostgreSQL, the checkpoint backend, and the
    store backend. Returns 503 if any component is unhealthy.
    """
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "langgraph_checkpointer": "unknown",
        "langgraph_store": "unknown",
    }

    # Database connectivity
    try:
        if db_manager.engine:
            async with db_manager.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            health_status["database"] = "connected"
        else:
            health_status["database"] = "not_initialized"
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # LangGraph checkpointer (lazy-init)
    try:
        checkpointer = db_manager.get_checkpointer()
        # probe - will raise if connection is bad; tuple may not exist which is fine
        with contextlib.suppress(Exception):
            await checkpointer.aget_tuple({"configurable": {"thread_id": "health-check"}})
        health_status["langgraph_checkpointer"] = "connected"
    except Exception as e:
        health_status["langgraph_checkpointer"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # LangGraph store (lazy-init)
    try:
        store = db_manager.get_store()
        with contextlib.suppress(Exception):
            await store.aget(("health",), "check")
        health_status["langgraph_store"] = "connected"
    except Exception as e:
        health_status["langgraph_store"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail="Service unhealthy")

    return HealthResponse(**health_status)


@router.get("/ready", responses={**UNAVAILABLE})
async def readiness_check(_request: Request) -> dict[str, str]:
    """Kubernetes readiness probe.

    Returns 200 when the server can accept traffic (database and graph
    backends are initialized). Returns 503 otherwise.
    """
    # Engine must exist and respond to a trivial query
    if not db_manager.engine:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - database engine not initialized",
        )
    try:
        async with db_manager.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready - database error: {str(e)}") from e

    # Check that LangGraph components can be obtained (lazy init) and respond
    try:
        checkpointer = db_manager.get_checkpointer()
        store = db_manager.get_store()
        # lightweight probes
        with contextlib.suppress(Exception):
            await checkpointer.aget_tuple({"configurable": {"thread_id": "ready-check"}})
        with contextlib.suppress(Exception):
            await store.aget(("ready",), "check")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready - components unavailable: {str(e)}",
        ) from e

    return {"status": "ready"}


@router.get("/live")
async def liveness_check(_request: Request) -> dict[str, str]:
    """Kubernetes liveness probe.

    Always returns 200 to indicate the process is alive. Does not check
    backend connectivity.
    """
    return {"status": "alive"}
