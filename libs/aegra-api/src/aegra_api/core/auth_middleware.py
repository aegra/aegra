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

    def _load_auth_instance(self) -> Auth | None:
        """Load the auth instance from config or fallback to hardcoded candidates.

        Resolution order:
        1. Load from aegra.json auth.path config
        2. If no auth file found, returns None (noop handled directly in authenticate())

        Returns:
            Auth instance or None if not found (noop handled in authenticate() method)
        """
        # 1. Try loading from config
        try:
            auth_config = load_auth_config()
            if auth_config and "path" in auth_config:
                auth_path = auth_config["path"]
                logger.info(f"Loading auth from config path: {auth_path}")
                auth_instance = self._load_from_path(auth_path)
                if auth_instance:
                    return auth_instance
                logger.warning(f"Failed to load auth from config path: {auth_path}")
        except Exception as e:
            logger.warning(f"Error loading auth config: {e}")

        logger.debug("No auth instance found from config")
        return None

    def _load_from_path(self, path: str) -> Auth | None:
        """Load auth instance from path in format './file.py:var' or 'module:var'.

        Relative paths are resolved from the config file directory.

        Args:
            path: Import path in format './file.py:variable' or 'module.path:variable'

        Returns:
            Auth instance or None if loading fails
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

            return self._load_from_file(file_path, var_name)

        # Handle module format: module.path
        return self._load_from_module(module_path, var_name)

    def _load_from_file(self, file_path: Path, var_name: str) -> Auth | None:
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

    def _load_from_module(self, module_path: str, var_name: str) -> Auth | None:
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
            logger.debug("No auth file configured, defaulting to noop (anonymous) authentication")
            # Return anonymous user when no auth is configured
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


@functools.lru_cache(maxsize=1)
def get_auth_backend() -> AuthenticationBackend:
    """
    Get authentication backend based on AUTH_TYPE environment variable.

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


@functools.lru_cache(maxsize=1)
def get_auth_instance() -> Auth | None:
    """Get cached Auth instance for use by other modules.

    Uses LRU cache to ensure only one Auth instance is loaded per process.
    This allows other modules to access the same Auth instance used by
    the middleware without re-loading it.

    Returns:
        Auth instance or None if not configured/found
    """
    backend = LangGraphAuthBackend()
    return backend.auth_instance
