"""Configuration management for Aegra settings (HTTP, auth, store, MCP)"""

import json
from pathlib import Path
from typing import Any, TypedDict

import structlog

from aegra_api.settings import settings

logger = structlog.get_logger(__name__)


class CorsConfig(TypedDict, total=False):
    """CORS configuration options"""

    allow_origins: list[str]
    allow_methods: list[str]
    allow_headers: list[str]
    allow_credentials: bool
    expose_headers: list[str]
    max_age: int


class HttpConfig(TypedDict, total=False):
    """HTTP configuration options for custom routes and protocol adapters"""

    app: str
    """Import path for custom Starlette/FastAPI app to mount"""
    enable_custom_route_auth: bool
    """Apply Aegra authentication dependency to custom routes (uses FastAPI dependencies, not middleware)"""
    cors: CorsConfig | None
    """Custom CORS configuration"""
    disable_mcp: bool
    """Disable MCP (Model Context Protocol) endpoint. Default: False (enabled)."""


class StoreIndexConfig(TypedDict, total=False):
    """Configuration for vector embeddings in store.

    Enables semantic similarity search using pgvector.
    See: https://github.com/aegra/aegra/issues/104
    """

    dims: int
    """Embedding vector dimensions (e.g., 1536 for OpenAI text-embedding-3-small)"""
    embed: str
    """Embedding model in format '<provider>:<model-id>'
    Examples:
    - openai:text-embedding-3-small (1536 dims)
    - openai:text-embedding-3-large (3072 dims)
    - bedrock:amazon.titan-embed-text-v2:0 (1024 dims)
    - cohere:embed-english-v3.0 (1024 dims)
    """
    fields: list[str] | None
    """JSON fields to embed. Defaults to ["$"] (entire document).
    Examples:
    - ["$"] - Embed entire document as one unit
    - ["text", "summary"] - Embed specific top-level fields
    - ["metadata.title", "content.text"] - JSON path notation
    """


class StoreConfig(TypedDict, total=False):
    """Store configuration options"""

    index: StoreIndexConfig | None
    """Vector index configuration for semantic search"""


class AuthConfig(TypedDict, total=False):
    """Auth configuration options."""

    path: str
    """Import path for auth handler in format './file.py:variable' or 'module:variable'.
    Examples:
    - './auth.py:auth' - Load 'auth' from auth.py in project root
    - './src/auth/firebase.py:auth' - Load from nested path
    - 'mypackage.auth:auth' - Load from installed package
    """
    disable_studio_auth: bool
    """Disable authentication for LangGraph Studio connections"""


class McpAuthConfig(TypedDict, total=False):
    """MCP auth provider configuration.

    Points to a Python file exporting a FastMCP auth provider.
    """

    path: str
    """Import path for MCP auth provider in format './file.py:variable' or 'module:variable'.
    Examples:
    - './mcp_auth.py:mcp_auth' - Load from mcp_auth.py
    - './src/auth/mcp.py:provider' - Load from nested path
    - 'mypackage.mcp_auth:provider' - Load from installed package
    """


class McpConfig(TypedDict, total=False):
    """MCP (Model Context Protocol) behavior configuration."""

    final_response_only: bool
    """Return only the last AI message instead of the full graph state. Default: False."""
    auth: McpAuthConfig | None
    """Auth provider config for spec-compliant MCP OAuth. Points to a Python file
    exporting a FastMCP auth provider."""


def _resolve_config_path() -> Path | None:
    """Resolve config file path using standard resolution order.

    Resolution order:
    1) AEGRA_CONFIG env var (if set and file exists)
    2) aegra.json in CWD
    3) langgraph.json in CWD (fallback for compatibility)

    Returns:
        Path to config file or None if not found
    """
    # 1) Env var override - only use if file actually exists
    if env_path := settings.app.AEGRA_CONFIG:
        path = Path(env_path)
        if path.exists():
            return path
        logger.warning(f"AEGRA_CONFIG={env_path!r} not found, falling back to config discovery")

    # 2) aegra.json if present
    aegra_path = Path("aegra.json")
    if aegra_path.exists():
        return aegra_path

    # 3) fallback to langgraph.json
    langgraph_path = Path("langgraph.json")
    if langgraph_path.exists():
        return langgraph_path

    return None


def load_config() -> dict[str, Any] | None:
    """Load full config file using standard resolution order.

    Returns:
        Full config dict or None if not found
    """
    config_path = _resolve_config_path()
    if not config_path:
        return None

    try:
        with config_path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Config file {config_path} does not contain a JSON object")
            return None
        return data
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return None


def load_http_config() -> HttpConfig | None:
    """Load HTTP config from aegra.json or langgraph.json.

    Uses standard config resolution order.

    Returns:
        HTTP configuration dict or None if not found
    """
    config = load_config()
    if config is None:
        return None

    http_config = config.get("http")
    if http_config:
        config_path = _resolve_config_path()
        logger.info(f"Loaded HTTP config from {config_path}")
        return http_config

    return None


def load_store_config() -> StoreConfig | None:
    """Load store config from aegra.json or langgraph.json.

    Uses standard config resolution order.

    Returns:
        Store configuration dict or None if not found
    """
    config = load_config()
    if config is None:
        return None

    store_config = config.get("store")
    if store_config:
        config_path = _resolve_config_path()
        logger.info(f"Loaded store config from {config_path}")
        return store_config

    return None


def load_auth_config() -> AuthConfig | None:
    """Load auth config from aegra.json or langgraph.json.

    Uses standard config resolution order.

    Returns:
        Auth configuration dict or None if not found
    """
    config = load_config()
    if config is None:
        return None

    auth_config = config.get("auth")
    if auth_config:
        config_path = _resolve_config_path()
        logger.info(f"Loaded auth config from {config_path}")
        return auth_config

    return None


def load_mcp_config() -> McpConfig | None:
    """Load MCP config from aegra.json or langgraph.json.

    Uses standard config resolution order.

    Returns:
        MCP configuration dict or None if not found
    """
    config = load_config()
    if config is None:
        return None

    mcp_config = config.get("mcp")
    if mcp_config:
        config_path = _resolve_config_path()
        logger.info(f"Loaded MCP config from {config_path}")
        return mcp_config

    return None


def get_config_dir() -> Path | None:
    """Get the directory containing the config file.

    This is used to resolve relative paths in the config file
    (graphs, http.app, auth.path) relative to the config location.

    Returns:
        Path to config directory or None if no config found
    """
    config_path = _resolve_config_path()
    if config_path and config_path.exists():
        return config_path.parent.resolve()
    return None
