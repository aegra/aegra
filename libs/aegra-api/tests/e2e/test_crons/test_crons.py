"""E2E tests for cron job endpoints.

Covers all six SDK operations consumed by ``CronClient``:
    create (stateless), create_for_thread, update, delete, search, count.

Tests run against a live server (CRON_ENABLED is not required; the scheduler
is only needed for scheduled firing — these tests only verify the CRUD API).
"""

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


async def _create_cron_via_http(payload: dict) -> dict:
    """Create a cron with raw HTTP when the SDK surface lags behind the API."""
    async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=10.0) as client:
        response = await client.post("/runs/crons", json=payload)
        response.raise_for_status()
        return response.json()


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
