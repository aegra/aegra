"""
Authentication middleware integration for Aegra.

This module integrates authentication system with FastAPI
using Starlette's AuthenticationMiddleware.
"""

import functools
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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


class AuthLoadError(RuntimeError):
    """``auth.path`` is set but cannot be loaded. Serving as anonymous would drop all access control."""


def split_import_path(path: str) -> tuple[Path | str, str]:
    """Split an ``auth`` import path into its target and variable name.

    The target is the resolved file of a ``./file.py:var`` path, relative paths
    resolving from the config directory, or the dotted module of ``module:var``.

    Raises:
        ValueError: If the path has no ``:`` before the variable name
    """
    if ":" not in path:
        raise ValueError(f"Invalid import path format (missing ':'): {path}")

    module_path, var_name = path.rsplit(":", 1)
    is_file_path = module_path.endswith(".py") or module_path.startswith("./") or module_path.startswith("../")
    if not is_file_path:
        return module_path, var_name

    file_path = Path(module_path)
    if not file_path.is_absolute():
        # Fallback to CWD if no config found
        file_path = (get_config_dir() or Path.cwd()) / file_path
    return file_path.resolve(), var_name


def import_auth_target(target: Path | str) -> ModuleType:
    """Import the module or file an ``auth`` path names, once per process.

    The auth handler and the cron user resolver often share a file. They see
    one module and its state when both paths name it by path, give it the same
    module name, or name it once by path and once as a module: a file the
    ``auth`` config also names as a module is imported as that module.
    """
    if isinstance(target, str):
        return importlib.import_module(target)
    module_name = _module_naming(target)
    if module_name is not None:
        return importlib.import_module(module_name)
    return _import_file(target)


def _module_naming(file_path: Path) -> str | None:
    """Return the module an ``auth`` config path names that imports from ``file_path``."""
    auth_config = load_auth_config() or {}
    for path in (auth_config.get("path"), auth_config.get("cron_user")):
        target = split_import_path(path)[0] if path and ":" in path else None
        if isinstance(target, str) and _module_file(target) == file_path:
            return target
    return None


def _module_file(module_name: str) -> Path | None:
    """Return the file ``module_name`` imports from, or None when Python cannot find one."""
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as e:
        # Finding it runs its parent packages; the loader of the path naming it reports the failure.
        logger.warning(f"Could not find module {module_name} named in the auth config: {e}")
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def _import_file(file_path: Path) -> ModuleType:
    """Import ``file_path`` once per process under a name derived from its stem."""
    module_name = f"auth_module_{file_path.stem}"
    loaded = sys.modules.get(module_name)
    if loaded is not None and getattr(loaded, "__file__", None) == str(file_path):
        return loaded

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # User code can raise anything at import time; a later load must not find the partial module.
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


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
        """The Auth instance from ``auth.path``, or None when no path is configured."""
        auth_config = load_auth_config()
        if not auth_config or "path" not in auth_config:
            logger.debug("No auth path configured")
            return None
        auth_path = auth_config["path"]
        logger.info(f"Loading auth from config path: {auth_path}")
        return self._load_from_path(auth_path)

    def _load_from_path(self, path: str) -> Auth:
        """Load auth from './file.py:var' or 'module:var'. Relative paths resolve from the config directory."""
        try:
            target, var_name = split_import_path(path)
        except ValueError as e:
            raise AuthLoadError(f"Invalid auth path format (missing ':'): {path}") from e

        if isinstance(target, Path):
            return self._load_from_file(target, var_name)
        return self._load_from_module(target, var_name)

    def _load_from_file(self, file_path: Path, var_name: str) -> Auth:
        if not file_path.exists():
            raise AuthLoadError(f"Auth file not found: {file_path}")

        if not file_path.is_file():
            raise AuthLoadError(f"Auth path is not a file: {file_path}")

        # User code can raise anything at import time.
        try:
            auth_module = import_auth_target(file_path)
        except Exception as e:
            raise AuthLoadError(f"Error loading auth from {file_path}: {e}") from e

        auth_instance = getattr(auth_module, var_name, None)
        if not isinstance(auth_instance, Auth):
            raise AuthLoadError(f"Variable '{var_name}' in {file_path} is not an Auth instance")

        logger.info(f"Successfully loaded auth instance from {file_path}:{var_name}")
        return auth_instance

    def _load_from_module(self, module_path: str, var_name: str) -> Auth:
        # User code can raise anything at import time, not only ImportError.
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            raise AuthLoadError(f"Error loading auth from module {module_path}: {e}") from e

        auth_instance = getattr(module, var_name, None)
        if not isinstance(auth_instance, Auth):
            raise AuthLoadError(f"Variable '{var_name}' in module {module_path} is not an Auth instance")

        logger.info(f"Successfully loaded auth instance from {module_path}:{var_name}")
        return auth_instance

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
                "is_authenticated": False,
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


@functools.lru_cache(maxsize=1)
def get_auth_backend() -> AuthenticationBackend:
    """
    Get authentication backend based on AUTH_TYPE environment variable.

    Cached so the auth module is loaded once at first use, not per-request.

    Returns:
        AuthenticationBackend instance
    """
    auth_type = settings.app.AUTH_TYPE

    if auth_type in ["noop", "custom"]:
        logger.debug(f"Using auth backend with type: {auth_type}")
        return LangGraphAuthBackend()
    else:
        logger.warning(f"Unknown AUTH_TYPE: {auth_type}, using noop")
        return LangGraphAuthBackend()


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
