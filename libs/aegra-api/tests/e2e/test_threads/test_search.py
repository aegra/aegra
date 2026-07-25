"""E2E tests for POST /threads/search sort and filter behavior."""

import asyncio
import uuid

import pytest
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client


async def _seed_three_threads(tag: str) -> list[str]:
    """Create three threads tagged with a unique marker, spaced in time.

    Returns thread ids in creation order (oldest first).
    """
    client = get_e2e_client()
    ids: list[str] = []
    for i in range(3):
        thread = await client.threads.create(metadata={"search_test_tag": tag, "seq": str(i)})
        ids.append(thread["thread_id"])
        # Force distinct created_at timestamps. 100ms leaves headroom for slow CI
        # runners where 50ms could collide on low-resolution clocks.
        await asyncio.sleep(0.1)
    elog(f"Seeded threads for tag {tag}", ids)
    return ids


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_order_by_created_at_asc_e2e() -> None:
    """order_by='created_at ASC' returns threads in creation order."""
    tag = f"sort-asc-{uuid.uuid4().hex[:8]}"
    created = await _seed_three_threads(tag)

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post(
            "/threads/search",
            json={"metadata": {"search_test_tag": tag}, "order_by": "created_at ASC", "limit": 100},
        )
    assert resp.status_code == 200, resp.text
    returned = [t["thread_id"] for t in resp.json()]
    elog("ASC result", returned)
    assert returned == created


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_order_by_created_at_desc_e2e() -> None:
    """order_by='created_at DESC' returns threads newest-first."""
    tag = f"sort-desc-{uuid.uuid4().hex[:8]}"
    created = await _seed_three_threads(tag)

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post(
            "/threads/search",
            json={"metadata": {"search_test_tag": tag}, "order_by": "created_at DESC", "limit": 100},
        )
    assert resp.status_code == 200, resp.text
    returned = [t["thread_id"] for t in resp.json()]
    elog("DESC result", returned)
    assert returned == list(reversed(created))


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_sdk_sort_by_asc_e2e() -> None:
    """SDK-style sort_by='created_at', sort_order='asc' sorts ascending."""
    tag = f"sdk-asc-{uuid.uuid4().hex[:8]}"
    created = await _seed_three_threads(tag)

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post(
            "/threads/search",
            json={
                "metadata": {"search_test_tag": tag},
                "sort_by": "created_at",
                "sort_order": "asc",
                "limit": 100,
            },
        )
    assert resp.status_code == 200, resp.text
    returned = [t["thread_id"] for t in resp.json()]
    elog("SDK ASC result", returned)
    assert returned == created


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_sort_by_takes_precedence_over_order_by_e2e() -> None:
    """When both sort_by and order_by are sent, sort_by wins."""
    tag = f"precedence-{uuid.uuid4().hex[:8]}"
    created = await _seed_three_threads(tag)

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        # sort_by without sort_order → defaults to desc. order_by asks for asc.
        # sort_by precedence means we should get desc order.
        resp = await http_client.post(
            "/threads/search",
            json={
                "metadata": {"search_test_tag": tag},
                "sort_by": "created_at",
                "order_by": "created_at ASC",
                "limit": 100,
            },
        )
    assert resp.status_code == 200, resp.text
    returned = [t["thread_id"] for t in resp.json()]
    elog("Precedence result", returned)
    assert returned == list(reversed(created)), "sort_by default-desc must beat order_by ASC"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_metadata_bool_filter_e2e() -> None:
    """metadata={'active': True} matches JSON-bool rows written via SDK."""
    tag = f"bool-{uuid.uuid4().hex[:8]}"
    client = get_e2e_client()

    active_ids: list[str] = []
    inactive_ids: list[str] = []
    for i in range(3):
        active = i != 1  # indices 0 and 2 are active
        thread = await client.threads.create(metadata={"search_test_tag": tag, "active": active})
        (active_ids if active else inactive_ids).append(thread["thread_id"])
        await asyncio.sleep(0.05)
    elog("Seeded bool threads", {"active": active_ids, "inactive": inactive_ids})

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post(
            "/threads/search",
            json={"metadata": {"search_test_tag": tag, "active": True}, "limit": 100},
        )
    assert resp.status_code == 200, resp.text
    returned = {t["thread_id"] for t in resp.json()}
    elog("active=True result", sorted(returned))
    assert returned == set(active_ids), f"expected {active_ids}, got {returned}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_pagination_stable_with_tied_sort_key_e2e() -> None:
    """Paginating by a non-unique sort key (status) must partition cleanly.

    Without a stable secondary sort, Postgres can return rows in arbitrary
    order within tied buckets, so two pages can overlap or drop rows. The
    handler now appends thread_id as a tie-break.
    """
    tag = f"pagination-{uuid.uuid4().hex[:8]}"
    client = get_e2e_client()

    seeded: list[str] = []
    for _ in range(6):
        thread = await client.threads.create(metadata={"search_test_tag": tag})
        seeded.append(thread["thread_id"])
    elog("Seeded threads (all status=idle, identical sort key)", seeded)

    page_size = 2
    collected: list[str] = []
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        for offset in range(0, len(seeded), page_size):
            resp = await http_client.post(
                "/threads/search",
                json={
                    "metadata": {"search_test_tag": tag},
                    "sort_by": "status",
                    "sort_order": "asc",
                    "limit": page_size,
                    "offset": offset,
                },
            )
            assert resp.status_code == 200, resp.text
            page = [t["thread_id"] for t in resp.json()]
            elog(f"Page offset={offset}", page)
            assert len(page) == page_size, f"page at offset {offset} had {len(page)} rows"
            collected.extend(page)

    assert sorted(collected) == sorted(seeded), (
        f"pagination dropped or duplicated rows: collected={collected}, seeded={seeded}"
    )
    assert len(collected) == len(set(collected)), f"pagination returned dupes: {collected}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_malformed_order_by_falls_back_e2e() -> None:
    """Unknown/malformed order_by must not 500 — falls back to default ordering."""
    tag = f"sort-bad-{uuid.uuid4().hex[:8]}"
    created = await _seed_three_threads(tag)

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        for bad in ["nonexistent_col", "password; DROP TABLE", ""]:
            resp = await http_client.post(
                "/threads/search",
                json={"metadata": {"search_test_tag": tag}, "order_by": bad, "limit": 100},
            )
            assert resp.status_code == 200, f"order_by={bad!r} → {resp.status_code}: {resp.text}"
            returned = {t["thread_id"] for t in resp.json()}
            assert returned == set(created), f"order_by={bad!r} dropped rows: {returned}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_select_and_extract_values_e2e() -> None:
    """select/extract return projected values from latest state in one search call."""
    client = get_e2e_client()
    tag = f"select-extract-{uuid.uuid4().hex[:8]}"
    thread = await client.threads.create(metadata={"search_test_tag": tag, "graph_id": "agent"})
    thread_id = thread["thread_id"]

    await client.threads.update_state(
        thread_id,
        values={
            "messages": [
                {"type": "human", "content": "what is rust?"},
                {"type": "ai", "content": "A systems language."},
            ]
        },
        as_node="__start__",
    )

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post(
            "/threads/search",
            json={
                "metadata": {"search_test_tag": tag},
                "select": ["thread_id"],
                "extract": {
                    "title": "values.messages[0].content",
                    "last_msg": "values.messages[-1].content",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["thread_id"] == thread_id
        assert "values" not in rows[0]
        extracted = rows[0]["extracted"]
        # Content may be plain string or structured depending on serializer
        title = extracted["title"]
        last = extracted["last_msg"]
        title_text = title if isinstance(title, str) else str(title)
        last_text = last if isinstance(last, str) else str(last)
        assert "rust" in title_text.lower()
        assert "systems" in last_text.lower() or "language" in last_text.lower()

        values_resp = await http_client.post(
            "/threads/search",
            json={
                "metadata": {"search_test_tag": tag},
                "select": ["thread_id", "values"],
            },
        )
        assert values_resp.status_code == 200, values_resp.text
        values_rows = values_resp.json()
        assert "values" in values_rows[0]
        assert "messages" in values_rows[0]["values"]
