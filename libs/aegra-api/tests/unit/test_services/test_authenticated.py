"""Unit tests for the Authenticated service base class."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from aegra_api.models.auth import User
from aegra_api.services.authenticated import Authenticated


class _Service(Authenticated):
    resource = "threads"


@pytest.fixture
def service() -> _Service:
    return _Service(session=AsyncMock(), user=User(identity="user-123"))


class TestDispatch:
    @pytest.mark.asyncio
    async def test_builds_context_with_resource_and_action(self, service: _Service) -> None:
        with patch(
            "aegra_api.services.authenticated.handle_event",
            new=AsyncMock(return_value=None),
        ) as handle_event:
            await service._dispatch("read", {"thread_id": "t-1"})

        ctx = handle_event.await_args.args[0]
        assert ctx.resource == "threads"
        assert ctx.action == "read"
        assert ctx.user.identity == "user-123"

    @pytest.mark.asyncio
    async def test_returns_handler_filters(self, service: _Service) -> None:
        filters = {"user_id": "user-123"}
        with patch(
            "aegra_api.services.authenticated.handle_event",
            new=AsyncMock(return_value=filters),
        ):
            result = await service._dispatch("search", {})

        assert result == filters

    @pytest.mark.asyncio
    async def test_propagates_handler_denial(self, service: _Service) -> None:
        with (
            patch(
                "aegra_api.services.authenticated.handle_event",
                new=AsyncMock(side_effect=HTTPException(status_code=403, detail="Forbidden")),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await service._dispatch("delete", {"thread_id": "t-1"})

        assert exc_info.value.status_code == 403
