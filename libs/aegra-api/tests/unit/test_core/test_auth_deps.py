"""Unit tests for auth dependencies"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.authentication import AuthCredentials, AuthenticationError

from aegra_api.core.auth_deps import (
    AuthenticatedUser,
    _to_user_model,
    auth_dependency,
    get_current_user,
    require_auth,
)
from aegra_api.core.auth_middleware import LangGraphUser
from aegra_api.models.auth import User


class TestRequireAuth:
    """Test require_auth dependency"""

    @pytest.mark.asyncio
    async def test_require_auth_success(self):
        """Test require_auth with successful authentication"""
        user_data = {
            "identity": "user-123",
            "display_name": "Test User",
            "is_authenticated": True,
            "email": "test@example.com",
        }
        credentials = AuthCredentials(["read", "write"])
        langgraph_user = LangGraphUser(user_data)

        mock_backend = Mock()
        mock_backend.authenticate = AsyncMock(return_value=(credentials, langgraph_user))

        mock_request = Mock(spec=Request)
        mock_request.scope = {}
        mock_request.user = None

        with patch("aegra_api.core.auth_deps.get_auth_backend", return_value=mock_backend):
            user = await require_auth(mock_request)

            assert isinstance(user, User)
            assert user.identity == "user-123"
            assert user.display_name == "Test User"
            assert user.email == "test@example.com"
            assert mock_request.scope["user"] == langgraph_user
            assert mock_request.scope["auth"] == credentials

    @pytest.mark.asyncio
    async def test_require_auth_no_result(self):
        """Test require_auth when backend returns None"""
        mock_backend = Mock()
        mock_backend.authenticate = AsyncMock(return_value=None)

        mock_request = Mock(spec=Request)
        mock_request.scope = {}

        with patch("aegra_api.core.auth_deps.get_auth_backend", return_value=mock_backend):
            with pytest.raises(HTTPException) as exc_info:
                await require_auth(mock_request)

            assert exc_info.value.status_code == 401
            assert "Authentication required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_auth_exception(self):
        """Test require_auth when backend raises exception"""
        mock_backend = Mock()
        mock_backend.authenticate = AsyncMock(side_effect=Exception("Auth error"))

        mock_request = Mock(spec=Request)
        mock_request.scope = {}

        with patch("aegra_api.core.auth_deps.get_auth_backend", return_value=mock_backend):
            with pytest.raises(HTTPException) as exc_info:
                await require_auth(mock_request)

            assert exc_info.value.status_code == 401
            assert "Auth error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_auth_preserves_custom_fields(self):
        """Test that require_auth preserves custom fields from auth handler"""
        user_data = {
            "identity": "user-123",
            "display_name": "Test User",
            "subscription_tier": "premium",
            "team_id": "team-456",
            "custom_metadata": {"key": "value"},
        }
        credentials = AuthCredentials([])
        langgraph_user = LangGraphUser(user_data)

        mock_backend = Mock()
        mock_backend.authenticate = AsyncMock(return_value=(credentials, langgraph_user))

        mock_request = Mock(spec=Request)
        mock_request.scope = {}
        mock_request.user = None

        with patch("aegra_api.core.auth_deps.get_auth_backend", return_value=mock_backend):
            user = await require_auth(mock_request)

            assert user.identity == "user-123"
            # Custom fields should be accessible
            assert hasattr(user, "subscription_tier")
            assert user.subscription_tier == "premium"
            assert hasattr(user, "team_id")
            assert user.team_id == "team-456"


class TestGetCurrentUser:
    """Test get_current_user legacy function"""

    def test_get_current_user_from_scope(self):
        """Test get_current_user reads from request.scope"""
        user_data = {
            "identity": "user-123",
            "display_name": "Test User",
        }
        langgraph_user = LangGraphUser(user_data)

        mock_request = Mock(spec=Request)
        mock_request.scope = {"user": langgraph_user}

        user = get_current_user(mock_request)

        assert isinstance(user, User)
        assert user.identity == "user-123"

    def test_get_current_user_from_request_user(self):
        """Test get_current_user falls back to request.user"""
        user_data = {
            "identity": "user-123",
            "display_name": "Test User",
        }
        langgraph_user = LangGraphUser(user_data)

        mock_request = Mock(spec=Request)
        mock_request.scope = {}
        mock_request.user = langgraph_user

        user = get_current_user(mock_request)

        assert isinstance(user, User)
        assert user.identity == "user-123"

    def test_get_current_user_not_authenticated(self):
        """Test get_current_user raises when user not found"""
        mock_request = Mock(spec=Request)
        mock_request.scope = {}
        mock_request.user = None

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(mock_request)

        assert exc_info.value.status_code == 401

    def test_get_current_user_preserves_auth_state(self):
        """Test get_current_user preserves is_authenticated when auth is informational."""
        user_data = {
            "identity": "user-123",
            "is_authenticated": False,
        }
        langgraph_user = LangGraphUser(user_data)

        mock_request = Mock(spec=Request)
        mock_request.scope = {"user": langgraph_user}

        user = get_current_user(mock_request)

        assert user.identity == "user-123"
        assert user.is_authenticated is False

    def test_noop_auth_server_still_serves_requests(self):
        """No-auth mode should still satisfy protected router dependencies."""
        app = FastAPI()
        router = APIRouter(dependencies=auth_dependency)

        @router.get("/whoami")
        async def whoami(user: User = Depends(get_current_user)) -> dict[str, object]:
            return {"identity": user.identity, "is_authenticated": user.is_authenticated}

        app.include_router(router)

        backend = Mock()
        anonymous_user = LangGraphUser(
            {
                "identity": "anonymous",
                "display_name": "Anonymous User",
                "is_authenticated": False,
            }
        )
        backend.authenticate = AsyncMock(return_value=(AuthCredentials([]), anonymous_user))

        with patch("aegra_api.core.auth_deps.get_auth_backend", return_value=backend):
            client = TestClient(app)
            response = client.get("/whoami")

        assert response.status_code == 200
        assert response.json() == {"identity": "anonymous", "is_authenticated": False}

    def test_auth_handler_exception_still_returns_401(self):
        """Auth handler failures should still reject protected requests."""
        app = FastAPI()
        router = APIRouter(dependencies=auth_dependency)

        @router.get("/whoami")
        async def whoami(user: User = Depends(get_current_user)) -> dict[str, str]:
            return {"identity": user.identity}

        app.include_router(router)

        backend = Mock()
        backend.authenticate = AsyncMock(side_effect=AuthenticationError("Invalid token"))

        with patch("aegra_api.core.auth_deps.get_auth_backend", return_value=backend):
            client = TestClient(app)
            response = client.get("/whoami")

        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid token"}


class TestToUserModel:
    """Test _to_user_model helper function"""

    def test_to_user_model_from_langgraph_user(self):
        """Test conversion from LangGraphUser"""
        user_data = {
            "identity": "user-123",
            "display_name": "Test User",
            "email": "test@example.com",
        }
        langgraph_user = LangGraphUser(user_data)

        user = _to_user_model(langgraph_user)

        assert isinstance(user, User)
        assert user.identity == "user-123"
        assert user.display_name == "Test User"
        assert user.email == "test@example.com"

    def test_to_user_model_from_dict(self):
        """Test conversion from dict"""
        user_data = {
            "identity": "user-123",
            "display_name": "Test User",
        }

        user = _to_user_model(user_data)

        assert isinstance(user, User)
        assert user.identity == "user-123"

    def test_to_user_model_defaults_display_name(self):
        """Test that display_name defaults to identity"""
        user_data = {"identity": "user-123"}

        user = _to_user_model(user_data)

        assert user.display_name == "user-123"

    def test_to_user_model_missing_identity(self):
        """Test that missing identity raises error"""
        user_data = {"display_name": "Test User"}

        with pytest.raises(HTTPException) as exc_info:
            _to_user_model(user_data)

        assert exc_info.value.status_code == 401


class TestTypeAliases:
    """Test type aliases and constants"""

    def test_authenticated_user_type(self):
        """Test AuthenticatedUser type alias exists"""
        # Just verify it's defined and is the right type
        assert AuthenticatedUser is not None

    def test_auth_dependency_list(self):
        """Test auth_dependency is a list of dependencies"""
        assert isinstance(auth_dependency, list)
        assert len(auth_dependency) == 1
