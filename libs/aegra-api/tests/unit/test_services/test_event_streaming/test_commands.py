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


async def _dispatch(payload: dict[str, Any], user: User, *, session: Any = None) -> tuple[dict, str | None]:
    return await cmd.handle_command(payload, session=session or AsyncMock(), thread_id="t1", user=user)


class TestRunStart:
    async def test_run_start_returns_run_id(self, prepared_run: AsyncMock, user: User) -> None:
        resp, run_id = await _dispatch(
            {"id": 1, "method": "run.start", "params": {"assistant_id": "agent", "input": {"messages": []}}},
            user,
        )
        assert resp == {
            "type": "success",
            "id": 1,
            "result": {"run_id": "run-xyz"},
            "meta": {"applied_through_seq": 0},
        }
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

    async def test_run_start_forwards_interrupt_breakpoints(self, prepared_run: AsyncMock, user: User) -> None:
        """interrupt_before/after must reach RunCreate so v2 clients can set HITL breakpoints."""
        await _dispatch(
            {
                "id": 1,
                "method": "run.start",
                "params": {
                    "assistant_id": "agent",
                    "input": {"x": 1},
                    "interrupt_before": ["node_a"],
                    "interrupt_after": "node_b",
                },
            },
            user,
        )
        request = prepared_run.call_args.args[2]
        assert request.interrupt_before == ["node_a"]
        assert request.interrupt_after == "node_b"

    async def test_run_start_missing_assistant_id_is_invalid(self, prepared_run: AsyncMock, user: User) -> None:
        resp, run_id = await _dispatch({"id": 1, "method": "run.start", "params": {"input": {}}}, user)
        assert resp["type"] == "error"
        assert resp["error"] == "invalid_argument"
        assert run_id is None
        prepared_run.assert_not_called()

    async def test_run_start_forwards_multitask_strategy(self, prepared_run: AsyncMock, user: User) -> None:
        await _dispatch(
            {
                "id": 1,
                "method": "run.start",
                "params": {"assistant_id": "agent", "input": {"x": 1}, "multitaskStrategy": "reject"},
            },
            user,
        )
        assert prepared_run.call_args.args[2].multitask_strategy == "reject"

    async def test_run_start_unknown_multitask_strategy_is_invalid(self, prepared_run: AsyncMock, user: User) -> None:
        resp, _ = await _dispatch(
            {
                "id": 1,
                "method": "run.start",
                "params": {"assistant_id": "agent", "input": {"x": 1}, "multitaskStrategy": "yolo"},
            },
            user,
        )
        assert resp["error"] == "invalid_argument"
        prepared_run.assert_not_called()

    async def test_run_start_on_interrupted_thread_resumes_with_input(
        self, prepared_run: AsyncMock, user: User
    ) -> None:
        """Input sent to an interrupted thread answers the interrupt instead of
        starting a fresh turn that would discard pending tasks."""
        session = AsyncMock()
        session.scalar = AsyncMock(return_value="interrupted")
        await _dispatch(
            {"id": 1, "method": "run.start", "params": {"assistant_id": "agent", "input": {"answer": 42}}},
            user,
            session=session,
        )
        request = prepared_run.call_args.args[2]
        assert request.command == {"resume": {"answer": 42}}
        assert request.input is None


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

    async def test_input_respond_forwards_metadata(self, prepared_run: AsyncMock, user: User) -> None:
        await _dispatch(
            {
                "id": 2,
                "method": "input.respond",
                "params": {"assistant_id": "agent", "response": "yes", "metadata": {"source": "review-ui"}},
            },
            user,
        )
        request = prepared_run.call_args.args[2]
        assert request.metadata == {"source": "review-ui"}

    async def test_input_respond_recovers_assistant_from_thread(self, prepared_run: AsyncMock, user: User) -> None:
        """The stock SDK sends no assistant_id; recover it from the thread's last run."""
        interrupt_id = "a" * 32
        session = AsyncMock()
        session.scalar = AsyncMock(return_value="bound-agent")
        resp, run_id = await _dispatch(
            {"id": 2, "method": "input.respond", "params": {"interrupt_id": interrupt_id, "response": "yes"}},
            user,
            session=session,
        )
        assert resp["type"] == "success"
        request = prepared_run.call_args.args[2]
        assert request.assistant_id == "bound-agent"
        # Targeted resume: id-keyed map so multiple pending interrupts route correctly.
        assert request.command == {"resume": {interrupt_id: "yes"}}

    async def test_input_respond_no_run_to_resume_is_error(self, prepared_run: AsyncMock, user: User) -> None:
        """No assistant_id and no prior run on the thread → on-protocol error, no run started."""
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)
        resp, run_id = await _dispatch(
            {"id": 2, "method": "input.respond", "params": {"interrupt_id": "b" * 32, "response": "yes"}},
            user,
            session=session,
        )
        assert resp["error"] == "no_such_run"
        assert run_id is None
        prepared_run.assert_not_called()

    async def test_input_respond_without_interrupt_id_resumes_untargeted(
        self, prepared_run: AsyncMock, user: User
    ) -> None:
        resp, _ = await _dispatch(
            {"id": 2, "method": "input.respond", "params": {"assistant_id": "agent", "response": "yes"}}, user
        )
        assert resp["type"] == "success"
        assert prepared_run.call_args.args[2].command == {"resume": "yes"}

    async def test_input_respond_malformed_interrupt_id_is_no_such_interrupt(
        self, prepared_run: AsyncMock, user: User
    ) -> None:
        """A non-interrupt-shaped id can never target a resume; erroring beats silently
        resuming whatever happens to be pending."""
        resp, run_id = await _dispatch(
            {"id": 2, "method": "input.respond", "params": {"interrupt_id": "garbage", "response": "yes"}}, user
        )
        assert resp["error"] == "no_such_interrupt"
        assert run_id is None
        prepared_run.assert_not_called()

    async def test_input_respond_batch_responses_merge_into_one_resume(
        self, prepared_run: AsyncMock, user: User
    ) -> None:
        """Parallel interrupts resume in a single command via the responses array."""
        id_a, id_b = "a" * 32, "b" * 32
        resp, _ = await _dispatch(
            {
                "id": 2,
                "method": "input.respond",
                "params": {
                    "assistant_id": "agent",
                    "responses": [
                        {"interrupt_id": id_a, "response": {"action": "approve"}},
                        {"interrupt_id": id_b, "response": [{"type": "ignore"}]},
                    ],
                },
            },
            user,
        )
        assert resp["type"] == "success"
        assert prepared_run.call_args.args[2].command == {
            "resume": {id_a: {"action": "approve"}, id_b: [{"type": "ignore"}]}
        }

    async def test_input_respond_empty_batch_is_invalid(self, prepared_run: AsyncMock, user: User) -> None:
        resp, _ = await _dispatch(
            {"id": 2, "method": "input.respond", "params": {"assistant_id": "agent", "responses": []}}, user
        )
        assert resp["error"] == "invalid_argument"
        prepared_run.assert_not_called()


class TestErrors:
    async def test_unknown_method_is_unknown_command(self, user: User) -> None:
        resp, run_id = await _dispatch({"id": 3, "method": "agent.getTree", "params": {}}, user)
        assert resp["error"] == "unknown_command"
        assert run_id is None

    async def test_unexpected_exception_wraps_as_unknown_error(
        self, monkeypatch: pytest.MonkeyPatch, user: User
    ) -> None:
        """A non-HTTP, non-validation crash must stay an on-protocol envelope, not a 500."""

        async def boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("db exploded")

        monkeypatch.setattr(cmd, "_prepare_run", boom)
        resp, run_id = await _dispatch(
            {"id": 9, "method": "run.start", "params": {"assistant_id": "x", "input": {}}}, user
        )
        assert resp["type"] == "error"
        assert resp["error"] == "unknown_error"
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
