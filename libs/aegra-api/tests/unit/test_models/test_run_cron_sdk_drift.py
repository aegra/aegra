"""Every JSON key ``langgraph_sdk`` can send for a run or a cron must be a declared field (#503).

Pydantic drops undeclared keys, so a field added by a new SDK release would
vanish without an error. This fails first and names the release.
"""

import inspect
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from operator import attrgetter
from typing import Any

import httpx
import pytest
from langgraph_sdk.client import LangGraphClient
from pydantic import BaseModel

from aegra_api.models.crons import CronCreate, CronUpdate
from aegra_api.models.runs import RunCreate

SDK_VERSION = version("langgraph-sdk")

_PATH_ARGUMENTS: dict[str, str] = {"thread_id": "thread-1", "cron_id": "cron-1"}
# Arguments the SDK keeps on the client side; everything else becomes a body key.
_CLIENT_SIDE_ARGUMENTS: frozenset[str] = frozenset({"headers", "params", "on_run_created", "raise_error", "version"})

# The SDK strips None before sending, so every argument needs a real value.
_SAMPLES: dict[str, Any] = {
    "assistant_id": "agent",
    "input": {"messages": []},
    "command": {"resume": "ok"},
    "stream_mode": ["values"],
    "stream_subgraphs": True,
    "stream_resumable": True,
    "metadata": {"source": "drift"},
    "config": {"configurable": {}},
    "context": {"tenant": "t"},
    "checkpoint": {"checkpoint_id": "c"},
    "checkpoint_id": "c",
    "checkpoint_during": True,
    "interrupt_before": ["node"],
    "interrupt_after": ["node"],
    "feedback_keys": ["quality"],
    "on_disconnect": "cancel",
    "on_completion": "keep",
    "on_run_completed": "keep",
    "webhook": "https://example.com/hook",
    "multitask_strategy": "reject",
    "if_not_exists": "create",
    "after_seconds": 1,
    "durability": "sync",
    "langsmith_tracing": {"project_name": "p"},
    "schedule": "* * * * *",
    "end_time": datetime.now(UTC) + timedelta(days=1),
    "enabled": True,
    "timezone": "UTC",
}
# An argument a future SDK adds is sent too, so its body key shows up here.
_UNKNOWN_ARGUMENT_SAMPLE = "x"


def _every_argument(method: Callable[..., Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for name in inspect.signature(method).parameters:
        if name in _PATH_ARGUMENTS:
            arguments[name] = _PATH_ARGUMENTS[name]
        elif name not in _CLIENT_SIDE_ARGUMENTS:
            arguments[name] = _SAMPLES.get(name, _UNKNOWN_ARGUMENT_SAMPLE)
    return arguments


async def _sent_body(method_path: str) -> dict[str, Any]:
    """Call the SDK method with every argument set and return the JSON body it sends."""
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"")
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(base_url="http://sdk.test", transport=httpx.MockTransport(respond)) as http:
        method = attrgetter(method_path)(LangGraphClient(http))
        result = method(**_every_argument(method))
        if inspect.isawaitable(result):
            await result
        else:
            async for _ in result:
                pass
    return bodies[0]


@pytest.mark.filterwarnings("ignore:`checkpoint_during` is deprecated:DeprecationWarning")
@pytest.mark.parametrize(
    ("method_path", "model"),
    [
        ("runs.create", RunCreate),
        ("runs.stream", RunCreate),
        ("runs.wait", RunCreate),
        ("crons.create", CronCreate),
        ("crons.create_for_thread", CronCreate),
        ("crons.update", CronUpdate),
    ],
)
async def test_every_sdk_body_key_is_a_declared_field(method_path: str, model: type[BaseModel]) -> None:
    body = await _sent_body(method_path)

    undeclared = sorted(set(body) - set(model.model_fields))

    assert not undeclared, (
        f"langgraph-sdk {SDK_VERSION} {method_path} sends {undeclared}, which {model.__name__} does not declare. "
        "Declare each key so it changes behavior, returns 422, or is a documented no-op."
    )
