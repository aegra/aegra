"""Pin v2 command handlers to the Agent Protocol param shapes.

The key sets below are copied from ``RunStartParams`` / ``InputRespondOne`` /
``InputRespondMany`` in the public protocol spec (langchain-ai/agent-protocol,
``streaming/js/protocol.ts``). When the spec grows a key, add it here first;
the tests then fail until the handler forwards it or lists it as a no-op.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from aegra_api.models import User
from aegra_api.services.event_streaming import commands as cmd

SPEC_RUN_START_KEYS: frozenset[str] = frozenset({"assistant_id", "input", "config", "metadata", "langsmith_tracer"})
SPEC_INPUT_RESPOND_ONE_KEYS: frozenset[str] = frozenset(
    {"namespace", "interrupt_id", "response", "update", "goto", "config", "metadata"}
)
SPEC_INPUT_RESPOND_MANY_KEYS: frozenset[str] = frozenset({"responses", "update", "goto", "config", "metadata"})

# Spec keys the server reads for validation only or has no backend for.
# Each one is documented in docs/guides/streaming.mdx.
DOCUMENTED_NOOP_KEYS: frozenset[str] = frozenset(
    {
        "langsmith_tracer",  # LangSmith-side routing; tracing here is OpenTelemetry.
        "namespace",  # interrupt ids are namespace hashes, so the id alone targets the resume.
    }
)


@pytest.fixture
def prepared_run(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=("run-1", object(), object()))
    monkeypatch.setattr(cmd, "_prepare_run", mock)
    return mock


async def _dispatch(method: str, params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    return await cmd.handle_command(
        {"id": 1, "method": method, "params": params}, session=AsyncMock(), thread_id="t1", user=User(identity="u1")
    )


class TestDeclaredKeysCoverSpec:
    def test_run_start_declares_every_spec_key(self) -> None:
        assert SPEC_RUN_START_KEYS <= cmd.RUN_START_KEYS

    def test_input_respond_declares_every_spec_key(self) -> None:
        assert SPEC_INPUT_RESPOND_ONE_KEYS | SPEC_INPUT_RESPOND_MANY_KEYS <= cmd.INPUT_RESPOND_KEYS


class TestEverySpecKeyChangesTheRun:
    """A spec key must either land on the RunCreate or be an explicit no-op."""

    async def test_run_start_forwards_all_spec_keys(self, prepared_run: AsyncMock) -> None:
        params: dict[str, Any] = {
            "assistant_id": "agent",
            "input": {"messages": [1]},
            "config": {"recursion_limit": 7},
            "metadata": {"source": "spec"},
            "langsmith_tracer": {"project_name": "p"},
        }
        assert set(params) == SPEC_RUN_START_KEYS

        resp, _ = await _dispatch("run.start", params)

        assert resp["type"] == "success"
        request = prepared_run.call_args.args[2]
        assert request.assistant_id == "agent"
        assert request.input == {"messages": [1]}
        assert request.config == {"recursion_limit": 7}
        assert request.metadata == {"source": "spec"}
        assert SPEC_RUN_START_KEYS - set(request.model_fields) <= DOCUMENTED_NOOP_KEYS

    async def test_input_respond_one_forwards_all_spec_keys(self, prepared_run: AsyncMock) -> None:
        params: dict[str, Any] = {
            "namespace": ["child"],
            "interrupt_id": "a" * 32,
            "response": {"ok": True},
            "update": {"flag": 1},
            "goto": "next",
            "config": {"recursion_limit": 7},
            "metadata": {"source": "spec"},
            "assistant_id": "agent",
        }
        assert set(params) - {"assistant_id"} == SPEC_INPUT_RESPOND_ONE_KEYS

        resp, _ = await _dispatch("input.respond", params)

        assert resp["type"] == "success"
        request = prepared_run.call_args.args[2]
        assert request.command == {"resume": {"a" * 32: {"ok": True}}, "update": {"flag": 1}, "goto": "next"}
        assert request.config == {"recursion_limit": 7}
        assert request.metadata == {"source": "spec"}

    async def test_input_respond_many_forwards_all_spec_keys(self, prepared_run: AsyncMock) -> None:
        id_a, id_b = "a" * 32, "b" * 32
        params: dict[str, Any] = {
            "responses": [{"interrupt_id": id_a, "response": 1}, {"interrupt_id": id_b, "response": 2}],
            "update": {"flag": 1},
            "goto": [{"node": "n", "input": {}}],
            "config": {"recursion_limit": 7},
            "metadata": {"source": "spec"},
            "assistant_id": "agent",
        }
        assert set(params) - {"assistant_id"} == SPEC_INPUT_RESPOND_MANY_KEYS

        resp, _ = await _dispatch("input.respond", params)

        assert resp["type"] == "success"
        request = prepared_run.call_args.args[2]
        assert request.command == {
            "resume": {id_a: 1, id_b: 2},
            "update": {"flag": 1},
            "goto": [{"node": "n", "input": {}}],
        }
        assert request.config == {"recursion_limit": 7}
        assert request.metadata == {"source": "spec"}

    def test_noop_keys_are_spec_keys(self) -> None:
        all_spec = SPEC_RUN_START_KEYS | SPEC_INPUT_RESPOND_ONE_KEYS | SPEC_INPUT_RESPOND_MANY_KEYS
        assert all_spec >= DOCUMENTED_NOOP_KEYS
