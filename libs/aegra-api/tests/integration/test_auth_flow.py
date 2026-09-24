"""Integration tests for authentication flow.

These tests validate:
1. Auth loading from config path
2. Mock JWT auth handler behavior
3. User model with custom fields
4. Noop auth fallback
5. Custom auth module is loaded once and reused across requests
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.authentication import AuthCredentials, AuthenticationError
from starlette.requests import HTTPConnection

from aegra_api.core import auth_middleware as auth_middleware_module
from aegra_api.core.auth_deps import require_auth
from aegra_api.core.auth_middleware import LangGraphAuthBackend, LangGraphUser, get_auth_backend
from aegra_api.models.auth import User


class TestAuthLoadingFromConfig:
    """Test loading auth handler from config file"""

    @pytest.fixture
    def mock_auth_file(self, tmp_path):
        """Create a temporary mock auth file"""
        auth_file = tmp_path / "test_auth.py"
        auth_file.write_text(
            """
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def authenticate(headers: dict) -> dict:
    return {
        "identity": "config-user",
        "display_name": "Config User",
        "is_authenticated": True,
        "permissions": ["read"],
    }
"""
        )
        return auth_file

    @pytest.fixture
    def aegra_config_with_auth(self, tmp_path, mock_auth_file, monkeypatch):
        """Create aegra.json with auth.path pointing to mock auth"""
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "aegra.json"
        config_file.write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "auth": {"path": f"./{mock_auth_file.name}:auth"},
                }
            )
        )
        return config_file

    def test_load_auth_from_config_path(self, aegra_config_with_auth, mock_auth_file):
        """Test that auth loads successfully from config path"""
        from aegra_api.config import load_auth_config
        from aegra_api.core.auth_middleware import LangGraphAuthBackend

        auth_config = load_auth_config()
        assert auth_config is not None
        assert "path" in auth_config

        backend = LangGraphAuthBackend()
        assert backend.auth_instance is not None
        assert backend.auth_instance._authenticate_handler is not None

    @pytest.mark.asyncio
    async def test_auth_handler_from_config_works(self, aegra_config_with_auth, mock_auth_file):
        """Test that auth handler loaded from config actually works"""
        backend = LangGraphAuthBackend()

        assert backend.auth_instance is not None

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer any-token"}

        credentials, user = await backend.authenticate(mock_conn)

        assert isinstance(user, LangGraphUser)
        assert user.identity == "config-user"
        assert user.display_name == "Config User"
        assert isinstance(credentials, AuthCredentials)


class TestMockJWTAuth:
    """Test mock JWT auth handler behavior"""

    @pytest.fixture
    def mock_jwt_auth_backend(self):
        """Create backend with mock JWT auth"""
        # Import the mock auth handler
        import sys

        # Add tests/fixtures to path
        fixtures_path = Path(__file__).parent.parent / "fixtures"
        if str(fixtures_path) not in sys.path:
            sys.path.insert(0, str(fixtures_path))

        from mock_jwt_auth import auth

        backend = LangGraphAuthBackend()
        backend.auth_instance = auth
        return backend

    @pytest.fixture
    def mock_jwt_auth_backend_from_config(self, tmp_path, monkeypatch):
        """Create backend by loading mock JWT auth from config path (realistic scenario)"""

        # Get the actual fixtures directory
        fixtures_path = Path(__file__).parent.parent / "fixtures"
        mock_auth_file = fixtures_path / "mock_jwt_auth.py"

        # Copy mock auth file to tmp_path
        import shutil

        test_auth_file = tmp_path / "mock_jwt_auth.py"
        shutil.copy(mock_auth_file, test_auth_file)

        # Create config pointing to it
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "aegra.json"
        config_file.write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "auth": {"path": f"./{test_auth_file.name}:auth"},
                }
            )
        )

        # Load backend - it should load from config
        backend = LangGraphAuthBackend()
        return backend

    @pytest.mark.asyncio
    async def test_mock_jwt_valid_token(self, mock_jwt_auth_backend):
        """Test mock JWT auth with valid token"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer mock-jwt-alice-admin-team123"}

        credentials, user = await mock_jwt_auth_backend.authenticate(mock_conn)

        assert isinstance(user, LangGraphUser)
        assert user.identity == "alice"
        assert user.display_name == "User alice"
        assert user.is_authenticated is True
        assert "admin" in credentials.scopes
        assert "admin:read" in credentials.scopes

    @pytest.mark.asyncio
    async def test_mock_jwt_custom_fields(self, mock_jwt_auth_backend):
        """Test that custom fields are preserved in user model"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer mock-jwt-bob-user-team456"}

        credentials, user = await mock_jwt_auth_backend.authenticate(mock_conn)

        # Check custom fields are accessible
        assert hasattr(user, "role")
        assert user.role == "user"
        assert hasattr(user, "subscription_tier")
        assert user.subscription_tier == "free"
        assert hasattr(user, "team_id")
        assert user.team_id == "team456"
        assert hasattr(user, "email")
        assert user.email == "bob@example.com"

    @pytest.mark.asyncio
    async def test_mock_jwt_invalid_token(self, mock_jwt_auth_backend):
        """Test mock JWT auth with invalid token"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer invalid-token"}

        with pytest.raises(AuthenticationError):
            await mock_jwt_auth_backend.authenticate(mock_conn)

    @pytest.mark.asyncio
    async def test_mock_jwt_missing_token(self, mock_jwt_auth_backend):
        """Test mock JWT auth with missing Authorization header"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {}

        with pytest.raises(AuthenticationError) as exc_info:
            await mock_jwt_auth_backend.authenticate(mock_conn)

        assert "Missing or invalid" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_mock_jwt_malformed_token(self, mock_jwt_auth_backend):
        """Test mock JWT auth with malformed token"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer mock-jwt-incomplete"}

        with pytest.raises(AuthenticationError) as exc_info:
            await mock_jwt_auth_backend.authenticate(mock_conn)

        assert "missing required fields" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_mock_jwt_premium_role(self, mock_jwt_auth_backend):
        """Test that premium role gets premium subscription tier"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer mock-jwt-charlie-premium-team789"}

        credentials, user = await mock_jwt_auth_backend.authenticate(mock_conn)

        assert user.role == "premium"
        assert user.subscription_tier == "premium"

    @pytest.mark.asyncio
    async def test_mock_jwt_loaded_from_config_path(self, mock_jwt_auth_backend_from_config):
        """Test that mock JWT auth can be loaded via config path mechanism"""
        backend = mock_jwt_auth_backend_from_config

        # Verify auth was loaded
        assert backend.auth_instance is not None
        assert backend.auth_instance._authenticate_handler is not None

        # Test it works
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer mock-jwt-testuser-admin-team123"}

        credentials, user = await backend.authenticate(mock_conn)

        assert user.identity == "testuser"
        assert user.role == "admin"
        assert user.subscription_tier == "premium"


class TestNoopAuth:
    """Test noop authentication fallback"""

    @pytest.mark.asyncio
    async def test_noop_auth_returns_anonymous_user(self):
        """Test that noop auth returns anonymous user when no auth configured"""
        backend = LangGraphAuthBackend()
        backend.auth_instance = None

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {}

        result = await backend.authenticate(mock_conn)

        assert result is not None
        credentials, user = result
        assert user.identity == "anonymous"
        assert user.display_name == "Anonymous User"
        assert user.is_authenticated is True
        assert isinstance(credentials, AuthCredentials)
        assert credentials.scopes == []


class TestUserModelCustomFields:
    """Test that custom fields flow through to User model"""

    @pytest.mark.asyncio
    async def test_user_model_preserves_custom_fields(self):
        """Test that custom fields from auth handler are accessible on User model"""
        import sys

        fixtures_path = Path(__file__).parent.parent / "fixtures"
        if str(fixtures_path) not in sys.path:
            sys.path.insert(0, str(fixtures_path))

        from mock_jwt_auth import auth

        backend = LangGraphAuthBackend()
        backend.auth_instance = auth

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer mock-jwt-dave-admin-team999"}
        mock_conn.scope = {}
        mock_conn.user = None

        # Authenticate
        credentials, langgraph_user = await backend.authenticate(mock_conn)

        # Convert to User model via require_auth dependency
        mock_request = Mock(spec=Request)
        mock_request.scope = {}
        mock_request.user = None

        with patch(
            "aegra_api.core.auth_deps.get_auth_backend_async",
            AsyncMock(return_value=backend),
        ):
            # Set scope with authenticated user
            mock_request.scope["user"] = langgraph_user
            mock_request.scope["auth"] = credentials

            from aegra_api.core.auth_deps import get_current_user

            user = get_current_user(mock_request)

            # Verify custom fields are accessible
            assert isinstance(user, User)
            assert user.identity == "dave"
            assert hasattr(user, "role")
            assert user.role == "admin"
            assert hasattr(user, "subscription_tier")
            assert user.subscription_tier == "premium"
            assert hasattr(user, "team_id")
            assert user.team_id == "team999"
            assert hasattr(user, "email")
            assert user.email == "dave@example.com"

    @pytest.mark.asyncio
    async def test_require_auth_with_mock_jwt(self):
        """Test require_auth dependency directly with mock JWT auth"""
        import sys

        fixtures_path = Path(__file__).parent.parent / "fixtures"
        if str(fixtures_path) not in sys.path:
            sys.path.insert(0, str(fixtures_path))

        from mock_jwt_auth import auth

        backend = LangGraphAuthBackend()
        backend.auth_instance = auth

        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer mock-jwt-eve-user-team111"}
        mock_request.scope = {}
        mock_request.user = None

        with patch(
            "aegra_api.core.auth_deps.get_auth_backend_async",
            AsyncMock(return_value=backend),
        ):
            user = await require_auth(mock_request)

            # Verify user is authenticated and custom fields are present
            assert isinstance(user, User)
            assert user.identity == "eve"
            assert user.is_authenticated is True
            assert hasattr(user, "role")
            assert user.role == "user"
            assert hasattr(user, "subscription_tier")
            assert user.subscription_tier == "free"
            assert hasattr(user, "team_id")
            assert user.team_id == "team111"

            # Verify request scope was set
            assert mock_request.scope["user"] is not None
            assert mock_request.scope["auth"] is not None


_CACHED_AUTH_SOURCE = """\
from langgraph_sdk import Auth

auth = Auth()


@auth.authenticate
async def authenticate(headers: dict) -> dict:
    return {
        "identity": "cached-user",
        "display_name": "Cached User",
        "is_authenticated": True,
        "permissions": ["read"],
        "team_id": "team-cache",
    }
"""


def _count_auth_file_specs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record each importlib spec created for a custom auth file."""
    real_spec_from_file_location = auth_middleware_module.importlib.util.spec_from_file_location
    locations: list[str] = []

    def counting_spec(
        name: str,
        location: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> ModuleSpec | None:
        if location is not None:
            locations.append(location)
        return real_spec_from_file_location(name, location, *args, **kwargs)

    monkeypatch.setattr(
        auth_middleware_module.importlib.util,
        "spec_from_file_location",
        counting_spec,
    )
    return locations


class TestAuthModuleCachedAcrossRequests:
    """Auth handler behavior must stay identical while the module loads once."""

    @pytest.fixture
    def cached_auth_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Project root with aegra.json pointing at a custom auth module."""
        auth_file = tmp_path / "cached_auth.py"
        auth_file.write_text(_CACHED_AUTH_SOURCE)
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "auth": {"path": "./cached_auth.py:auth"},
                }
            )
        )
        monkeypatch.chdir(tmp_path)
        return auth_file

    @pytest.mark.asyncio
    async def test_require_auth_loads_module_once_and_returns_same_user(
        self, cached_auth_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Several require_auth calls import the auth file once and keep handler results."""
        locations = _count_auth_file_specs(monkeypatch)
        users: list[User] = []
        for _ in range(4):
            mock_request = Mock(spec=Request)
            mock_request.headers = {"authorization": "Bearer unused"}
            mock_request.scope = {}
            mock_request.user = None
            users.append(await require_auth(mock_request))

        assert [user.identity for user in users] == ["cached-user"] * 4
        assert all(user.display_name == "Cached User" for user in users)
        assert all(user.team_id == "team-cache" for user in users)
        assert all(user.is_authenticated is True for user in users)
        assert locations == [str(cached_auth_project.resolve())]
        assert get_auth_backend() is get_auth_backend()

    def test_http_requests_reuse_cached_auth(self, cached_auth_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP-level requests must not re-import the auth module."""
        locations = _count_auth_file_specs(monkeypatch)

        app = FastAPI()

        @app.get("/whoami")
        async def whoami_ok(user: User = Depends(require_auth)) -> dict[str, str]:
            return {"identity": user.identity, "team_id": str(user.team_id)}

        client = TestClient(app)
        bodies = [client.get("/whoami").json() for _ in range(3)]

        assert bodies == [{"identity": "cached-user", "team_id": "team-cache"}] * 3
        assert locations == [str(cached_auth_project.resolve())]
