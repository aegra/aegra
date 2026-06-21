"""Tests for v2 command dispatch (run.start, input.respond, errors)."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from aegra_api.models import User
from aegra_api.services.event_streaming import commands as cmd


@pytest.fixture
def user() -> User:
    return User(identity="u1")


@pytest.fixture
def prepared_run(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Stub _prepare_run to return a fixed run_id without touching the DB."""
    mock = AsyncMock(return_value=("run-xyz", object(), object()))
    monkeypatch.setattr(cmd, "_prepare_run", mock)
    return mock


async def _dispatch(payload: dict[str, Any], user: User) -> tuple[dict, str | None]:
    return await cmd.handle_command(payload, session=AsyncMock(), thread_id="t1", user=user)


class TestRunStart:
    async def test_run_start_returns_run_id(self, prepared_run: AsyncMock, user: User) -> None:
        resp, run_id = await _dispatch(
            {"id": 1, "method": "run.start", "params": {"assistant_id": "agent", "input": {"messages": []}}},
            user,
        )
        assert resp == {"type": "success", "id": 1, "result": {"run_id": "run-xyz"}}
        assert run_id == "run-xyz"

    async def test_run_start_builds_runcreate(self, prepared_run: AsyncMock, user: User) -> None:
        await _dispatch(
            {
                "id": 1,
                "method": "run.start",
                "params": {"assistant_id": "agent", "input": {"x": 1}, "config": {"c": 2}},
            },
            user,
        )
        request = prepared_run.call_args.args[2]
        assert request.assistant_id == "agent"
        assert request.input == {"x": 1}
        assert request.config == {"c": 2}
        # v2 runs are flagged for the native v3 stream path.
        assert prepared_run.call_args.kwargs["event_streaming_v2"] is True

    async def test_run_start_missing_assistant_id_is_invalid(self, prepared_run: AsyncMock, user: User) -> None:
        resp, run_id = await _dispatch({"id": 1, "method": "run.start", "params": {"input": {}}}, user)
        assert resp["type"] == "error"
        assert resp["error"] == "invalid_argument"
        assert run_id is None
        prepared_run.assert_not_called()


class TestInputRespond:
    async def test_input_respond_resumes_with_command(self, prepared_run: AsyncMock, user: User) -> None:
        resp, run_id = await _dispatch(
            {"id": 2, "method": "input.respond", "params": {"assistant_id": "agent", "response": "yes"}},
            user,
        )
        assert resp["type"] == "success"
        request = prepared_run.call_args.args[2]
        assert request.command == {"resume": "yes"}

    async def test_input_respond_missing_response_is_invalid(self, prepared_run: AsyncMock, user: User) -> None:
        resp, _ = await _dispatch({"id": 2, "method": "input.respond", "params": {"assistant_id": "agent"}}, user)
        assert resp["error"] == "invalid_argument"


class TestErrors:
    async def test_unknown_method_is_not_supported(self, user: User) -> None:
        resp, run_id = await _dispatch({"id": 3, "method": "agent.getTree", "params": {}}, user)
        assert resp["error"] == "not_supported"
        assert run_id is None

    async def test_non_integer_id_is_invalid(self, user: User) -> None:
        resp, _ = await _dispatch({"id": "x", "method": "run.start", "params": {}}, user)
        assert resp == {"type": "error", "id": None, "error": "invalid_argument", "message": resp["message"]}

    async def test_non_dict_params_is_invalid(self, user: User) -> None:
        resp, _ = await _dispatch({"id": 1, "method": "run.start", "params": "nope"}, user)
        assert resp["error"] == "invalid_argument"

    async def test_prepare_http_404_maps_to_protocol_error(self, monkeypatch: pytest.MonkeyPatch, user: User) -> None:
        """An HTTPException from run prep returns an on-protocol error, not FastAPI's detail."""
        from fastapi import HTTPException

        async def boom(*_a: Any, **_k: Any) -> None:
            raise HTTPException(404, "Assistant 'x' not found")

        monkeypatch.setattr(cmd, "_prepare_run", boom)
        resp, run_id = await _dispatch(
            {"id": 5, "method": "run.start", "params": {"assistant_id": "x", "input": {}}}, user
        )
        assert resp == {"type": "error", "id": 5, "error": "no_such_run", "message": "Assistant 'x' not found"}
        assert run_id is None

    async def test_prepare_http_403_maps_to_permission_denied(
        self, monkeypatch: pytest.MonkeyPatch, user: User
    ) -> None:
        from fastapi import HTTPException

        async def boom(*_a: Any, **_k: Any) -> None:
            raise HTTPException(403, "nope")

        monkeypatch.setattr(cmd, "_prepare_run", boom)
        resp, _ = await _dispatch({"id": 6, "method": "run.start", "params": {"assistant_id": "x", "input": {}}}, user)
        assert resp["error"] == "permission_denied"

    async def test_malformed_runcreate_params_map_to_invalid_argument(self, user: User) -> None:
        """RunCreate validation failure (no input/command/checkpoint) is on-protocol."""
        resp, run_id = await _dispatch({"id": 7, "method": "run.start", "params": {"assistant_id": "x"}}, user)
        assert resp["type"] == "error"
        assert resp["error"] == "invalid_argument"
        assert run_id is None
