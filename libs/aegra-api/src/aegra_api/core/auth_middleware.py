"""
Authentication middleware integration for Aegra.

This module integrates authentication system with FastAPI
using Starlette's AuthenticationMiddleware.
"""

import asyncio
import functools
import importlib
import importlib.util
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Protocol

import structlog
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import MinimalUserDict
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    BaseUser,
)
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from aegra_api.config import get_config_dir, load_auth_config
from aegra_api.models.errors import AgentProtocolError
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Thread callers (lru_cache concurrent misses). Never acquired on a running loop.
_auth_thread_lock = threading.RLock()
# Shared in-flight backend fill. Callers join through get_auth_backend().
_auth_fill_guard = threading.Lock()
_auth_backend_fill: Future[AuthenticationBackend] | None = None


class _LruCachedFn[T](Protocol):
    def __call__(self) -> T: ...
    def cache_info(self) -> Any: ...


def _lru_fill_once[T](cached_fn: _LruCachedFn[T]) -> T:
    """Single-flight an lru_cache fill without blocking a running event loop."""
    if cached_fn.cache_info().currsize:
        return cached_fn()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        with _auth_thread_lock:
            return cached_fn()
    return cached_fn()


class LangGraphUser(BaseUser):
    """
    User wrapper that implements Starlette's BaseUser interface
    while preserving auth data.
    """

    def __init__(self, user_data: Auth.types.MinimalUserDict):
        self._user_data = user_data

    @property
    def identity(self) -> str:
        return self._user_data["identity"]

    @property
    def is_authenticated(self) -> bool:
        return self._user_data.get("is_authenticated", True)

    @property
    def display_name(self) -> str:
        return self._user_data.get("display_name", self.identity)

    def __getattr__(self, name: str) -> Any:
        """Allow access to any additional fields from auth data"""
        if name in self._user_data:
            return self._user_data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def to_dict(self) -> MinimalUserDict:
        """Return the underlying user data dict"""
        return self._user_data.copy()


def _load_auth_from_file(file_path: Path, var_name: str) -> Auth | None:
    """Load auth instance from a file path.

    Args:
        file_path: Path to the Python file
        var_name: Name of the variable to load

    Returns:
        Auth instance or None if loading fails
    """
    try:
        if not file_path.exists():
            logger.warning(f"Auth file not found: {file_path}")
            return None

        if not file_path.is_file():
            logger.warning(f"Auth path is not a file: {file_path} (is directory: {file_path.is_dir()})")
            return None

        # Create a unique module name based on the file path
        module_name = f"auth_module_{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            logger.error(f"Could not load auth module from {file_path}")
            return None

        auth_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = auth_module
        spec.loader.exec_module(auth_module)

        auth_instance = getattr(auth_module, var_name, None)
        if not isinstance(auth_instance, Auth):
            logger.error(f"Variable '{var_name}' in {file_path} is not an Auth instance")
            return None

        logger.info(f"Successfully loaded auth instance from {file_path}:{var_name}")
        return auth_instance

    except Exception as e:
        logger.error(f"Error loading auth from {file_path}: {e}", exc_info=True)
        return None


def _load_auth_from_module(module_path: str, var_name: str) -> Auth | None:
    """Load auth instance from an installed module.

    Args:
        module_path: Dotted module path (e.g., 'mypackage.auth')
        var_name: Name of the variable to load

    Returns:
        Auth instance or None if loading fails
    """
    try:
        module = importlib.import_module(module_path)
        auth_instance = getattr(module, var_name, None)

        if not isinstance(auth_instance, Auth):
            logger.error(f"Variable '{var_name}' in module {module_path} is not an Auth instance")
            return None

        logger.info(f"Successfully loaded auth instance from {module_path}:{var_name}")
        return auth_instance

    except ImportError as e:
        logger.error(f"Could not import module {module_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading auth from {module_path}: {e}", exc_info=True)
        return None


@functools.lru_cache(maxsize=32)
def _load_auth_from_path(path: str) -> Auth | None:
    """Load Auth from './file.py:var' or 'module:var'. Cached per config path.

    `aegra dev` reload starts a new process, so this does not hot-reload on disk edits.
    """
    if ":" not in path:
        logger.error(f"Invalid auth path format (missing ':'): {path}")
        return None

    module_path, var_name = path.rsplit(":", 1)

    # Handle file path format: ./file.py or ./path/to/file.py or ../file.py
    is_file_path = module_path.endswith(".py") or module_path.startswith("./") or module_path.startswith("../")
    if is_file_path:
        file_path = Path(module_path)

        # Resolve relative paths from config directory
        if not file_path.is_absolute():
            config_dir = get_config_dir()
            if config_dir:
                file_path = (config_dir / file_path).resolve()
            else:
                # Fallback to CWD if no config found
                file_path = (Path.cwd() / file_path).resolve()

        return _load_auth_from_file(file_path, var_name)

    # Handle module format: module.path
    return _load_auth_from_module(module_path, var_name)


@functools.lru_cache(maxsize=1)
def _load_auth_from_config_cached() -> Auth | None:
    """Read auth.path from config and load. Cached for the process lifetime."""
    try:
        auth_config = load_auth_config()
        if auth_config and "path" in auth_config:
            auth_path = auth_config["path"]
            logger.info(f"Loading auth from config path: {auth_path}")
            auth_instance = _load_auth_from_path(auth_path)
            if auth_instance:
                return auth_instance
            logger.warning(f"Failed to load auth from config path: {auth_path}")
    except Exception as e:
        logger.warning(f"Error loading auth config: {e}")

    logger.debug("No auth instance found from config")
    return None


def _cached_load_auth_from_config() -> Auth | None:
    """Load Auth from aegra.json; concurrent first-misses share one read."""
    return _lru_fill_once(_load_auth_from_config_cached)


class LangGraphAuthBackend(AuthenticationBackend):
    """
    Authentication backend that uses the auth system.

    This bridges @auth.authenticate handlers with
    Starlette's AuthenticationMiddleware.
    """

    def __init__(self) -> None:
        self.auth_instance = self._load_auth_instance()
        if self.auth_instance is None:
            logger.warning(
                "No auth file configured — all requests share a single 'anonymous' identity. "
                "Data is NOT isolated between users in this mode. "
                "Configure 'auth.path' in aegra.json before serving multiple users. "
                "See: https://docs.aegra.dev/guides/authentication"
            )

    def _load_auth_instance(self) -> Auth | None:
        """Load the auth instance from config or fallback to hardcoded candidates.

        Resolution order:
        1. Load from aegra.json auth.path config
        2. If no auth file found, returns None (noop handled directly in authenticate())

        Returns:
            Auth instance or None if not found (noop handled in authenticate() method)
        """
        return _cached_load_auth_from_config()

    async def authenticate(self, conn: HTTPConnection) -> tuple[AuthCredentials, BaseUser] | None:
        """
        Authenticate request using the configured auth system.

        Args:
            conn: HTTP connection containing request headers

        Returns:
            Tuple of (credentials, user) if authenticated, None otherwise

        Raises:
            AuthenticationError: If authentication fails
        """
        # Handle noop auth when no auth instance is configured
        # Default to noop (anonymous) authentication when no auth file is found,
        # regardless of AUTH_TYPE setting. This ensures the server works out-of-the-box.
        if self.auth_instance is None:
            # The no-auth warning is emitted once at startup in __init__;
            # logging here would repeat it on every request.
            logger.debug("No auth file configured, defaulting to noop (anonymous) authentication")
            # Return anonymous user when no auth is configured.
            # WARNING: all callers share this identity; no tenant isolation is enforced.
            user_data: Auth.types.MinimalUserDict = {
                "identity": "anonymous",
                "display_name": "Anonymous User",
                "is_authenticated": True,
            }
            credentials = AuthCredentials([])
            user = LangGraphUser(user_data)
            return credentials, user

        if self.auth_instance._authenticate_handler is None:
            logger.warning("No authenticate handler configured, skipping authentication")
            return None

        try:
            # Convert headers to dict format expected by auth handlers
            headers = {
                key.decode() if isinstance(key, bytes) else key: value.decode() if isinstance(value, bytes) else value
                for key, value in conn.headers.items()
            }

            # Call the authenticate handler
            user_data = await self.auth_instance._authenticate_handler(headers)

            if not user_data or not isinstance(user_data, dict):
                raise AuthenticationError("Invalid user data returned from auth handler")

            if "identity" not in user_data:
                raise AuthenticationError("Auth handler must return 'identity' field")

            # Extract permissions for credentials
            permissions = user_data.get("permissions", [])
            if isinstance(permissions, str):
                permissions = [permissions]

            # Create Starlette-compatible user and credentials
            credentials = AuthCredentials(permissions)
            user = LangGraphUser(user_data)

            logger.debug(f"Successfully authenticated user: {user.identity}")
            return credentials, user

        except Auth.exceptions.HTTPException as e:
            logger.warning(f"Authentication failed: {e.detail}")
            raise AuthenticationError(e.detail) from e

        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}", exc_info=True)
            raise AuthenticationError("Authentication system error") from e


def _claim_auth_backend_fill() -> tuple[Future[AuthenticationBackend], bool]:
    """Return the in-flight fill future and whether the caller must populate it."""
    global _auth_backend_fill
    with _auth_fill_guard:
        if _get_auth_backend_cached.cache_info().currsize:
            done: Future[AuthenticationBackend] = Future()
            done.set_result(_get_auth_backend_cached())
            return done, False
        if _auth_backend_fill is None or _auth_backend_fill.done():
            _auth_backend_fill = Future()
            return _auth_backend_fill, True
        return _auth_backend_fill, False


def _fail_auth_backend_fill(fut: Future[AuthenticationBackend], exc: BaseException) -> None:
    """Publish a joinable error; never attach CancelledError to waiters."""
    if fut.done():
        return
    if isinstance(exc, Exception):
        fut.set_exception(exc)
        return
    fut.set_exception(RuntimeError("auth backend initialization failed"))


def _complete_auth_backend_fill(fut: Future[AuthenticationBackend]) -> AuthenticationBackend:
    """Run the cached constructor and publish the result to joiners."""
    try:
        result = _get_auth_backend_cached()
    except BaseException as exc:
        _fail_auth_backend_fill(fut, exc)
        raise
    if not fut.done():
        fut.set_result(result)
    return result


@functools.lru_cache(maxsize=1)
def _get_auth_backend_cached() -> AuthenticationBackend:
    auth_type = settings.app.AUTH_TYPE

    if auth_type in ["noop", "custom"]:
        logger.debug(f"Using auth backend with type: {auth_type}")
        return LangGraphAuthBackend()
    else:
        logger.warning(f"Unknown AUTH_TYPE: {auth_type}, using noop")
        return LangGraphAuthBackend()


def get_auth_backend() -> AuthenticationBackend:
    """
    Get authentication backend based on AUTH_TYPE environment variable.

    Cached so the auth module is loaded once at first use, not per-request.

    Returns:
        AuthenticationBackend instance
    """
    if _get_auth_backend_cached.cache_info().currsize:
        return _get_auth_backend_cached()
    fut, owner = _claim_auth_backend_fill()
    if not owner:
        return fut.result()
    return _complete_auth_backend_fill(fut)


async def get_auth_backend_async() -> AuthenticationBackend:
    """Join get_auth_backend() off-loop so a sync waiter cannot deadlock this loop."""
    if _get_auth_backend_cached.cache_info().currsize:
        return _get_auth_backend_cached()
    # shield: cancelling one waiter must not cancel the worker running the shared fill.
    return await asyncio.shield(asyncio.to_thread(get_auth_backend))


def _clear_auth_loader_caches() -> None:
    """Drop process-level auth loader caches. Tests call this for isolation."""
    global _auth_backend_fill
    with _auth_fill_guard:
        _load_auth_from_config_cached.cache_clear()
        _load_auth_from_path.cache_clear()
        _get_auth_backend_cached.cache_clear()
        _auth_backend_fill = None


def on_auth_error(conn: HTTPConnection, exc: AuthenticationError) -> JSONResponse:
    """
    Handle authentication errors in Agent Protocol format.

    Args:
        conn: HTTP connection
        exc: Authentication error

    Returns:
        JSON response with Agent Protocol error format
    """
    logger.warning(f"Authentication error for {conn.url}: {exc}")

    return JSONResponse(
        status_code=401,
        content=AgentProtocolError(
            error="unauthorized",
            message=str(exc),
            details={"authentication_required": True},
        ).model_dump(),
    )


def get_auth_instance() -> Auth | None:
    """Get the Auth instance from the cached backend.

    Delegates to get_auth_backend() so only one Auth instance exists
    per process.

    Returns:
        Auth instance or None if not configured/found
    """
    backend = get_auth_backend()
    if isinstance(backend, LangGraphAuthBackend):
        return backend.auth_instance
    logger.warning(
        "get_auth_instance() called but backend is not LangGraphAuthBackend: %s",
        type(backend).__name__,
    )
    return None
