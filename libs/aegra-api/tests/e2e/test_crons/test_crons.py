"""E2E tests for cron job endpoints.

Covers all six SDK operations consumed by ``CronClient``:
    create (stateless), create_for_thread, update, delete, search, count.

Tests run against a live server (CRON_ENABLED is not required; the scheduler
is only needed for scheduled firing; most tests verify the CRUD API while
``test_cron_example_seconds_schedule_fires_on_live_scheduler`` exercises the
real scheduler when fast cron settings are enabled.
"""

import asyncio
from typing import Any
from uuid import uuid4

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client


def _extract_message_content(cron: dict) -> str | None:
    """Return the first message content stored in the cron payload, if present."""
    messages = cron.get("payload", {}).get("input", {}).get("messages", [])
    if not messages:
        return None
    return messages[0].get("content")


async def _find_cron_id_by_message(*, client, assistant_id: str, message: str) -> str:
    """Find exactly one cron for an assistant by its unique input marker."""
    crons = await client.crons.search(assistant_id=assistant_id)
    matches = [cron for cron in crons if _extract_message_content(cron) == message]
    assert len(matches) == 1
    return matches[0]["cron_id"]


async def _post_json(path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    """POST JSON to the live server and return status + parsed body."""
    async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=10.0) as client:
        response = await client.post(path, json=payload)
        try:
            body = response.json()
        except ValueError:
            body = {"text": response.text}
        return response.status_code, body


async def _create_cron_via_http(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a cron with raw HTTP when the SDK surface lags behind the API."""
    status, body = await _post_json("/runs/crons", payload)
    assert status == 200, body
    return body


async def _create_thread_cron_via_http(thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a thread-bound cron with raw HTTP."""
    status, body = await _post_json(f"/threads/{thread_id}/runs/crons", payload)
    assert status == 200, body
    return body


async def _search_crons_via_http(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Search crons with raw HTTP so tests can use fields the SDK may not expose yet."""
    status, body = await _post_json("/runs/crons/search", payload)
    assert status == 200, body
    assert isinstance(body, list)
    return body


def _tick_count(state: dict[str, Any]) -> int:
    """Count cron_example tick messages in a thread state payload."""
    messages = state.get("values", {}).get("messages", [])
    count = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, str) and content.startswith("Tick #"):
            count += 1
    return count


async def _wait_for_tick_count(
    *,
    client: Any,
    thread_id: str,
    minimum: int,
    attempts: int = 30,
) -> dict[str, Any]:
    """Poll thread state until cron_example has produced enough ticks."""
    latest_state: dict[str, Any] = {}
    for _ in range(attempts):
        latest_state = await client.threads.get_state(thread_id=thread_id)
        if _tick_count(latest_state) >= minimum:
            return latest_state
        await asyncio.sleep(1)
    pytest.fail(f"Expected at least {minimum} cron_example ticks, saw {_tick_count(latest_state)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_accepts_graph_id_as_assistant_id() -> None:
    """Create a cron using a graph ID directly and verify default assistant resolution."""
    client = get_e2e_client()
    marker = f"cron-via-graph-id-{uuid4()}"

    cron_run = await client.crons.create(
        "agent",
        schedule="0 2 * * *",
        input={"messages": [{"role": "user", "content": marker}]},
    )
    elog("Cron.create (graph id)", cron_run)

    assert "run_id" in cron_run
    assert cron_run["assistant_id"] != "agent"

    crons = await client.crons.search(assistant_id="agent")
    matching = [
        cron
        for cron in crons
        if cron.get("payload", {}).get("input", {}).get("messages", [{}])[0].get("content") == marker
    ]
    assert len(matching) == 1
    cron_id = matching[0]["cron_id"]

    await client.crons.delete(cron_id)
    elog("Cron deleted", {"cron_id": cron_id})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_stateless_create_and_delete() -> None:
    """Create a stateless cron, verify the first Run is returned, then delete it."""
    client = get_e2e_client()
    marker = f"cron-stateless-{uuid4()}"

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["e2e-cron-stateless"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]
    elog("Assistant", assistant)

    cron_run = await client.crons.create(
        assistant_id,
        schedule="0 3 * * *",  # 03:00 UTC every day — won't fire during test
        input={"messages": [{"role": "user", "content": marker}]},
    )
    elog("Cron.create (stateless)", cron_run)

    assert "run_id" in cron_run
    cron_id = await _find_cron_id_by_message(client=client, assistant_id=assistant_id, message=marker)

    await client.crons.delete(cron_id)
    elog("Cron deleted", {"cron_id": cron_id})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_disabled_create_returns_cron_without_first_run() -> None:
    """Creating with enabled=False persists the cron and suppresses the initial Run."""
    client = get_e2e_client()
    marker = f"cron-disabled-{uuid4()}"

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["e2e-cron-disabled"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    created = await _create_cron_via_http(
        {
            "assistant_id": assistant_id,
            "schedule": "0 21 * * *",
            "enabled": False,
            "input": {"messages": [{"role": "user", "content": marker}]},
        }
    )
    elog("Cron.create disabled", created)

    assert "cron_id" in created
    assert "run_id" not in created
    assert created["enabled"] is False
    assert _extract_message_content(created) == marker

    await client.crons.delete(created["cron_id"])
    elog("Cron deleted", {"cron_id": created["cron_id"]})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_for_thread_create_and_delete() -> None:
    """Create a thread-bound cron, verify the Run is returned, then delete."""
    client = get_e2e_client()
    marker = f"cron-thread-{uuid4()}"

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["e2e-cron-thread"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    elog("Thread", thread)

    cron_run = await client.crons.create_for_thread(
        thread_id,
        assistant_id,
        schedule="0 4 * * *",  # 04:00 UTC every day
        input={"messages": [{"role": "user", "content": marker}]},
    )
    elog("Cron.create_for_thread", cron_run)

    assert "run_id" in cron_run
    cron_id = await _find_cron_id_by_message(client=client, assistant_id=assistant_id, message=marker)

    await client.crons.delete(cron_id)
    elog("Cron deleted", {"cron_id": cron_id})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_delete_cascades_thread_cron() -> None:
    """Deleting a thread removes its thread-bound crons through the real database FK."""
    client = get_e2e_client()
    marker = f"cron-cascade-{uuid4()}"

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["e2e-cron-cascade"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    created = await _create_thread_cron_via_http(
        thread_id,
        {
            "assistant_id": assistant_id,
            "schedule": "0 22 * * *",
            "enabled": False,
            "input": {"messages": [{"role": "user", "content": marker}]},
        },
    )
    cron_id = created["cron_id"]
    elog("Thread cron created", created)

    before = await _search_crons_via_http({"assistant_id": assistant_id, "thread_id": thread_id})
    assert {cron["cron_id"] for cron in before} == {cron_id}

    await client.threads.delete(thread_id)
    after = await _search_crons_via_http({"assistant_id": assistant_id, "thread_id": thread_id})
    elog("Thread cron search after thread delete", after)
    assert after == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_search_and_count() -> None:
    """Create two crons, search and count them by assistant_id, then clean up."""
    client = get_e2e_client()
    marker_a = f"cron-search-a-{uuid4()}"
    marker_b = f"cron-search-b-{uuid4()}"

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["e2e-cron-search"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    run_a = await client.crons.create(
        assistant_id,
        schedule="0 5 * * *",
        input={"messages": [{"role": "user", "content": marker_a}]},
    )
    run_b = await client.crons.create(
        assistant_id,
        schedule="0 6 * * *",
        input={"messages": [{"role": "user", "content": marker_b}]},
    )
    assert "run_id" in run_a
    assert "run_id" in run_b
    cron_id_a = await _find_cron_id_by_message(client=client, assistant_id=assistant_id, message=marker_a)
    cron_id_b = await _find_cron_id_by_message(client=client, assistant_id=assistant_id, message=marker_b)
    elog("Created crons", {"a": cron_id_a, "b": cron_id_b})

    try:
        # search
        crons = await client.crons.search(assistant_id=assistant_id)
        elog("Cron.search", crons)
        found_ids = {c["cron_id"] for c in crons}
        assert cron_id_a in found_ids
        assert cron_id_b in found_ids

        # count
        total = await client.crons.count(assistant_id=assistant_id)
        elog("Cron.count", total)
        assert total >= 2
    finally:
        await client.crons.delete(cron_id_a)
        await client.crons.delete(cron_id_b)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_update() -> None:
    """Create a cron, update its schedule and enabled flag, verify the response."""
    client = get_e2e_client()
    marker = f"cron-update-{uuid4()}"

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["e2e-cron-update"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    cron_run = await client.crons.create(
        assistant_id,
        schedule="0 7 * * *",
        input={"messages": [{"role": "user", "content": marker}]},
    )
    assert "run_id" in cron_run
    cron_id = await _find_cron_id_by_message(client=client, assistant_id=assistant_id, message=marker)
    elog("Created cron", {"cron_id": cron_id})

    try:
        updated = await client.crons.update(
            cron_id,
            schedule="0 8 * * *",
            enabled=False,
        )
        elog("Cron.update", updated)

        assert updated["cron_id"] == cron_id
        assert updated["schedule"] == "0 8 * * *"
        assert updated["enabled"] is False
    finally:
        await client.crons.delete(cron_id)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"assistant_id": "missing", "schedule": "*/30 * * * * *"},
        {"assistant_id": "missing", "schedule": "0 1 * * *", "webhook": "file:///tmp/hook"},
        {"assistant_id": "missing", "schedule": "0 1 * * *", "end_time": "2000-01-01T00:00:00Z"},
    ],
)
async def test_cron_review_guards_reject_bad_http_requests(payload: dict[str, Any]) -> None:
    """New validation/feature gates reject bad requests before persistence."""
    if payload["schedule"].count(" ") == 5 and settings.cron.CRON_ALLOW_SECONDS_SCHEDULE:
        pytest.skip("seconds schedules are enabled in this environment")

    status, body = await _post_json("/runs/crons", payload)
    elog("Rejected cron create", {"status": status, "body": body})
    assert status == 422


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_with_timezone() -> None:
    """Create a cron with an IANA timezone; verify the timezone is stored in payload."""
    client = get_e2e_client()
    marker = f"cron-timezone-{uuid4()}"

    assistant = await client.assistants.create(
        graph_id="agent",
        config={"tags": ["e2e-cron-timezone"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    cron_run = await _create_cron_via_http(
        {
            "assistant_id": assistant_id,
            "schedule": "0 9 * * *",
            "timezone": "America/New_York",
            "input": {"messages": [{"role": "user", "content": marker}]},
        }
    )
    assert "run_id" in cron_run
    cron_id = await _find_cron_id_by_message(client=client, assistant_id=assistant_id, message=marker)
    elog("Created cron with timezone", {"cron_id": cron_id})

    try:
        # Verify via search
        crons = await client.crons.search(assistant_id=assistant_id)
        matching = [c for c in crons if c["cron_id"] == cron_id]
        assert len(matching) == 1
        stored_payload = matching[0].get("payload", {})
        elog("Stored payload", stored_payload)
        assert stored_payload.get("timezone") == "America/New_York"
    finally:
        await client.crons.delete(cron_id)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_example_seconds_schedule_fires_on_live_scheduler() -> None:
    """Run the documented cron_example graph through the real scheduler loop."""
    if not settings.cron.CRON_ALLOW_SECONDS_SCHEDULE:
        pytest.skip("requires CRON_ALLOW_SECONDS_SCHEDULE=true")
    if settings.cron.CRON_POLL_INTERVAL_SECONDS > 2:
        pytest.skip("requires CRON_POLL_INTERVAL_SECONDS<=2 for a fast smoke test")

    client = get_e2e_client()
    assistant = await client.assistants.create(
        graph_id="cron_example",
        config={"tags": ["e2e-cron-example-scheduler"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    cron_id: str | None = None

    try:
        cron_run = await _create_thread_cron_via_http(
            thread_id,
            {
                "assistant_id": assistant_id,
                "schedule": "*/5 * * * * *",
                "input": {"messages": []},
            },
        )
        cron_id = (await _search_crons_via_http({"assistant_id": assistant_id, "thread_id": thread_id}))[0]["cron_id"]
        elog("cron_example scheduled cron", {"cron_id": cron_id, "run": cron_run})

        await client.runs.join(thread_id, cron_run["run_id"])
        await _wait_for_tick_count(client=client, thread_id=thread_id, minimum=1, attempts=10)

        state = await _wait_for_tick_count(client=client, thread_id=thread_id, minimum=2, attempts=30)
        elog("cron_example state after scheduler fire", state)
        assert _tick_count(state) >= 2
    finally:
        if cron_id is not None:
            await client.crons.delete(cron_id)
        await client.threads.delete(thread_id)
