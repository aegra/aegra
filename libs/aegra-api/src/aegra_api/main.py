"""FastAPI application for Aegra (Agent Protocol Server)"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute, APIRouter

from aegra_api import __version__
from aegra_api.api.assistants import router as assistants_router
from aegra_api.api.runs import router as runs_router
from aegra_api.api.stateless_runs import router as stateless_runs_router
from aegra_api.api.store import router as store_router
from aegra_api.api.threads import router as threads_router
from aegra_api.config import CorsConfig, HttpConfig, get_config_dir, load_http_config
from aegra_api.core.app_loader import load_custom_app
from aegra_api.core.auth_deps import auth_dependency
from aegra_api.core.database import db_manager
from aegra_api.core.health import router as health_router
from aegra_api.core.migrations import run_migrations_async
from aegra_api.core.redis_manager import redis_manager
from aegra_api.core.route_merger import (
    merge_exception_handlers,
    merge_lifespans,
)
from aegra_api.middleware import ContentTypeFixMiddleware, StructLogMiddleware
from aegra_api.models.errors import AgentProtocolError, get_error_type
from aegra_api.observability.metrics import setup_prometheus_metrics
from aegra_api.observability.setup import setup_observability
from aegra_api.services.broker import broker_manager
from aegra_api.services.executor import executor
from aegra_api.services.js_bridge import stop_js_bridge
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.lease_reaper import lease_reaper
from aegra_api.settings import settings
from aegra_api.utils.setup_logging import setup_logging

OPENAPI_TAGS: list[dict[str, Any]] = [
    {"name": "Assistants", "description": "A configured instance of a graph."},
    {"name": "Threads", "description": "Accumulated state and outputs from a group of runs."},
    {"name": "Thread Runs", "description": "Invoke a graph on a thread, updating its persistent state."},
    {"name": "Stateless Runs", "description": "Invoke a graph without state or memory persistence."},
    {"name": "Store", "description": "Persistent key-value and semantic storage available from any thread."},
    {"name": "Health", "description": "Server health checks and service information."},
]

setup_logging()
logger = structlog.getLogger(__name__)

# Default CORS headers required for LangGraph SDK stream reconnection
DEFAULT_EXPOSE_HEADERS = ["Content-Location", "Location"]


def _log_connection_help(error: Exception) -> None:
    """Log a helpful error message when database connection fails."""
    logger.error(
        "Could not connect to PostgreSQL",
        error=str(error),
        hint="Check your database configuration and ensure PostgreSQL is running.",
    )
    logger.error(
        "Troubleshooting tips:\n"
        "  - Local development?  Run 'aegra dev' (starts PostgreSQL automatically)\n"
        "  - Docker deployment?  Run 'aegra up' (starts PostgreSQL + app)\n"
        "  - External database?  Check DATABASE_URL or POSTGRES_* vars in your .env\n"
        "  - Missing .env file?  Copy .env.example to .env and configure it"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for startup/shutdown"""
    # Auto-apply pending database migrations before anything else
    try:
        await run_migrations_async()
    except (ConnectionRefusedError, OSError) as e:
        _log_connection_help(e)
        raise

    # Startup: Initialize database and LangGraph components
    try:
        await db_manager.initialize()
    except (ConnectionRefusedError, OSError) as e:
        _log_connection_help(e)
        raise

    # Observability
    setup_observability()

    # Initialize LangGraph service
    langgraph_service = get_langgraph_service()
    await langgraph_service.initialize()

    # Initialize Redis broker (if enabled)
    if settings.redis.REDIS_BROKER_ENABLED:
        try:
            await redis_manager.initialize()
        except (ConnectionError, OSError) as e:
            logger.error(
                "Cannot connect to Redis. "
                "Set REDIS_BROKER_ENABLED=false for single-instance mode without Redis, "
                "or ensure Redis is running at REDIS_URL.",
                redis_url=settings.redis.REDIS_URL,
                error=str(e),
            )
            raise
    else:
        logger.warning(
            "Running without Redis broker. Background runs have no crash recovery "
            "or horizontal scaling. Set REDIS_BROKER_ENABLED=true and configure "
            "REDIS_URL for production use.",
        )

    # Start broker manager (cleanup task for in-memory, cancel listener for Redis)
    await broker_manager.start()

    # Start executor (spawns worker coroutines when Redis is enabled)
    await executor.start()

    # Start lease reaper (recovers crashed worker runs, Redis mode only)
    if settings.redis.REDIS_BROKER_ENABLED:
        await lease_reaper.start()

    yield

    # Shutdown order: reaper → executor (drains jobs) → broker → JS bridge → Redis → DB
    if settings.redis.REDIS_BROKER_ENABLED:
        await lease_reaper.stop()
    await executor.stop()
    await broker_manager.stop()

    # Stop JS bridge subprocess (if running)
    await stop_js_bridge()

    # Close Redis broker (if enabled)
    if settings.redis.REDIS_BROKER_ENABLED:
        await redis_manager.close()

    await db_manager.close()


# Define core exception handlers
async def agent_protocol_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTP exceptions to Agent Protocol error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=AgentProtocolError(
            error=get_error_type(exc.status_code),
            message=exc.detail,
            details=getattr(exc, "details", None),
        ).model_dump(),
    )


async def general_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    return JSONResponse(
        status_code=500,
        content=AgentProtocolError(
            error="internal_error",
            message="An unexpected error occurred",
            details={"exception": str(exc)},
        ).model_dump(),
    )


exception_handlers = {
    HTTPException: agent_protocol_exception_handler,
    Exception: general_exception_handler,
}


# Define root endpoint handler
async def root_handler() -> dict[str, str]:
    """Root endpoint"""
    return {
        "message": settings.app.PROJECT_NAME,
        "version": __version__,
        "status": "running",
    }


def _apply_auth_to_routes(app: FastAPI, auth_deps: list[Any]) -> None:
    """Apply auth dependency to all existing routes in the FastAPI app.

    This function recursively processes all routes including nested routers,
    adding the auth dependency to each route that doesn't already have it.
    Auth dependencies are prepended to ensure they run first (fail-fast).

    Args:
        app: FastAPI application instance
        auth_deps: List of dependencies to apply (e.g., [Depends(require_auth)])
    """

    def process_routes(routes: list) -> None:
        """Recursively process routes and nested routers."""
        for route in routes:
            if isinstance(route, APIRoute):
                # Add auth dependency if not already present
                existing_deps = list(route.dependencies or [])
                # Check if auth dependency is already present
                auth_dep_ids = {id(dep) for dep in auth_deps}
                existing_dep_ids = {id(dep) for dep in existing_deps}
                if not auth_dep_ids.intersection(existing_dep_ids):
                    # Prepend auth deps so they run first (fail-fast)
                    route.dependencies = auth_deps + existing_deps
            elif isinstance(route, APIRouter):
                # Process nested router
                process_routes(route.routes)
            elif hasattr(route, "routes"):
                # Handle other route types that have nested routes
                process_routes(route.routes)

    process_routes(app.routes)
    logger.info("Applied authentication dependency to custom routes")


def _add_cors_middleware(app: FastAPI, cors_config: CorsConfig | None) -> None:
    """Add CORS middleware with config or defaults.

    When ``allow_origins`` is ``["*"]`` (the default), ``allow_credentials``
    defaults to ``False`` because the combination of a wildcard origin with
    credentials is insecure — it allows any site to make credentialed requests.
    To enable ``allow_credentials``, specify concrete origins.

    Args:
        app: FastAPI application instance
        cors_config: CORS configuration dict or None for defaults
    """
    if cors_config:
        origins = cors_config.get("allow_origins", ["*"])
        credentials = cors_config.get(
            "allow_credentials",
            origins not in (["*"], "*"),
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=credentials,
            allow_methods=cors_config.get("allow_methods", ["*"]),
            allow_headers=cors_config.get("allow_headers", ["*"]),
            expose_headers=cors_config.get("expose_headers", DEFAULT_EXPOSE_HEADERS),
            max_age=cors_config.get("max_age", 600),
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=DEFAULT_EXPOSE_HEADERS,
        )


def _add_common_middleware(app: FastAPI, cors_config: CorsConfig | None) -> None:
    """Add common middleware stack in correct order.

    Middleware runs in reverse registration order, so we register:
    1. ContentTypeFixMiddleware (outermost - fixes text/plain → application/json)
    2. CORSMiddleware (handles preflight early)
    3. CorrelationIdMiddleware (adds request ID)
    4. StructLogMiddleware (innermost - logs with correlation ID)

    Args:
        app: FastAPI application instance
        cors_config: CORS configuration dict or None for defaults
    """
    app.add_middleware(StructLogMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    _add_cors_middleware(app, cors_config)
    app.add_middleware(ContentTypeFixMiddleware)


def _include_core_routers(app: FastAPI) -> None:
    """Include all core API routers with auth dependency.

    Routers are included in consistent order:
    1. Health (no auth)
    2. Assistants (with auth)
    3. Threads (with auth)
    4. Runs (with auth)
    5. Stateless Runs (with auth)
    6. Store (with auth)

    Args:
        app: FastAPI application instance
    """
    app.include_router(health_router)
    app.include_router(assistants_router)
    app.include_router(threads_router)
    app.include_router(runs_router)
    app.include_router(stateless_runs_router)
    app.include_router(store_router)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    http_config: HttpConfig | None = load_http_config()
    cors_config: CorsConfig | None = http_config.get("cors") if http_config else None

    # Try to load custom app if configured
    user_app = None
    if http_config and http_config.get("app"):
        try:
            config_dir = get_config_dir()
            user_app = load_custom_app(http_config["app"], base_dir=config_dir)
            logger.info("Custom app loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load custom app: {e}", exc_info=True)
            raise

    if user_app:
        if not isinstance(user_app, FastAPI):
            raise TypeError(
                "Custom apps must be FastAPI applications. Use: from fastapi import FastAPI; app = FastAPI()"
            )

        application = user_app
        if not application.openapi_tags:
            application.openapi_tags = OPENAPI_TAGS
        _include_core_routers(application)

        # Add root endpoint if not already defined
        if not any(route.path == "/" for route in application.routes if hasattr(route, "path")):
            application.get("/")(root_handler)

        application = merge_lifespans(application, lifespan)
        application = merge_exception_handlers(application, exception_handlers)
        _add_common_middleware(application, cors_config)

        # Apply auth to custom routes if enabled
        if http_config and http_config.get("enable_custom_route_auth", False):
            _apply_auth_to_routes(application, auth_dependency)
    else:
        application = FastAPI(
            title=settings.app.PROJECT_NAME,
            description="Production-ready Agent Protocol server",
            version=settings.app.VERSION,
            debug=settings.app.DEBUG,
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=lifespan,
            openapi_tags=OPENAPI_TAGS,
        )

        _add_common_middleware(application, cors_config)
        _include_core_routers(application)

        for exc_type, handler in exception_handlers.items():
            application.exception_handler(exc_type)(handler)

        application.get("/")(root_handler)

    setup_prometheus_metrics(application)

    return application


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(settings.app.PORT)
    uvicorn.run(app, host=settings.app.HOST, port=port)  # nosec B104 - binding to all interfaces is intentional
