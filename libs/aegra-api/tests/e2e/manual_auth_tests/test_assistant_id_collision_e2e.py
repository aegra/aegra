"""E2E tests for creating an assistant whose ID another user already holds.

``assistant_pkey`` is global while every ownership read is scoped to the caller,
so this collision is invisible to the create handler's own SELECT. It used to
reach Postgres as an unconditional INSERT and come back as an unhandled
``UniqueViolationError`` — a 500. It must answer 409, and must never hand the
caller a row they do not own.

Require an auth-enabled server. See README.md for setup.
Run with: make e2e-auth  or  pytest -m auth_only
"""

import uuid

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


def get_server_url() -> str:
    return settings.app.SERVER_URL


def get_auth_headers(user_id: str, role: str = "user", team_id: str = "team1") -> dict[str, str]:
    token = f"mock-jwt-{user_id}-{role}-{team_id}"
    return {"Authorization": f"Bearer {token}"}


async def _create_as(user_id: str, payload: dict[str, object]) -> httpx.Response:
    async with httpx.AsyncClient(base_url=get_server_url(), headers=get_auth_headers(user_id), timeout=30.0) as http:
        return await http.post("/assistants", json=payload)


@pytest.mark.e2e
@pytest.mark.auth_only
class TestAssistantIdOwnedByAnotherUser:
    """A taken assistant_id conflicts for everyone else, whatever if_exists says."""

    @pytest.mark.parametrize("if_exists", ["error", "do_nothing"])
    @pytest.mark.asyncio
    async def test_returns_409_not_500(self, if_exists: str) -> None:
        assistant_id = f"cross-user-{uuid.uuid4()}"

        alice = await _create_as(
            "alice",
            {"assistant_id": assistant_id, "graph_id": "agent", "config": {"tags": [assistant_id]}},
        )
        assert alice.status_code == 200, alice.text

        bob = await _create_as(
            "bob",
            {
                "assistant_id": assistant_id,
                "graph_id": "agent",
                "config": {"tags": [f"bob-{assistant_id}"]},
                "if_exists": if_exists,
            },
        )

        elog("Bob create on Alice's assistant_id", {"if_exists": if_exists, "status": bob.status_code})
        assert bob.status_code == 409, f"expected 409, got {bob.status_code}: {bob.text}"

    @pytest.mark.asyncio
    async def test_does_not_hand_over_the_other_users_assistant(self) -> None:
        """do_nothing must not turn a foreign row into a successful read."""
        assistant_id = f"cross-user-{uuid.uuid4()}"

        alice = await _create_as(
            "alice",
            {
                "assistant_id": assistant_id,
                "graph_id": "agent",
                "config": {"tags": [assistant_id]},
                "metadata": {"owner_marker": "alice"},
            },
        )
        assert alice.status_code == 200, alice.text

        bob = await _create_as(
            "bob",
            {"assistant_id": assistant_id, "graph_id": "agent", "if_exists": "do_nothing"},
        )
        assert bob.status_code == 409, f"expected 409, got {bob.status_code}: {bob.text}"
        assert "owner_marker" not in bob.text, "Alice's assistant leaked into Bob's response"

        # Alice's assistant is untouched by the failed create
        async with httpx.AsyncClient(
            base_url=get_server_url(), headers=get_auth_headers("alice"), timeout=30.0
        ) as http:
            fetched = await http.get(f"/assistants/{assistant_id}")
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["metadata"]["owner_marker"] == "alice"
