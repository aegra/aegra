"""Integration tests for cron CRUD API endpoints.

Uses the same test-client + mocked-service pattern as test_assistants_crud.py.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from aegra_api.models.crons import CronResponse
from aegra_api.services.cron_service import get_cron_service
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.test_helpers import make_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = "2025-06-01T00:00:00+00:00"


def _cron_response(**overrides: Any) -> CronResponse:
    """Build a CronResponse with sensible defaults."""
    defaults = {
        "cron_id": "cron-001",
        "assistant_id": "asst-001",
        "thread_id": None,
        "schedule": "*/5 * * * *",
        "payload": {"input": {"x": 1}},
        "metadata": {"env": "test"},
        "user_id": "test-user",
        "enabled": True,
        "on_run_completed": None,
        "end_time": None,
        "next_run_date": _NOW,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    defaults.update(overrides)
    return CronResponse(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cron_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_cron_service: AsyncMock):
    """TestClient with the crons router mounted and service mocked."""
    app = create_test_app(include_runs=False, include_threads=False)

    from aegra_api.api import crons as crons_module
    from aegra_api.core.orm import get_session

    app.include_router(crons_module.router)
    app.dependency_overrides[get_cron_service] = lambda: mock_cron_service

    # Override get_session so create endpoints don't require a real DB
    async def _mock_session():
        yield AsyncMock()

    app.dependency_overrides[get_session] = _mock_session

    with (
        patch("aegra_api.api.crons.handle_event", new_callable=AsyncMock) as mock_handle,
        patch("aegra_api.api.crons._trigger_first_run", new_callable=AsyncMock) as mock_trigger,
    ):
        mock_handle.return_value = None
        mock_trigger.return_value = make_run()
        yield make_client(app)


# ---------------------------------------------------------------------------
# POST /runs/crons  →  Run
# ---------------------------------------------------------------------------


class TestCreateCron:
    """Test POST /runs/crons (stateless create)."""

    def test_creates_cron_and_returns_run(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/runs/crons",
            json={
                "assistant_id": "asst-001",
                "schedule": "*/5 * * * *",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        mock_cron_service.create_cron.assert_called_once()

    def test_passes_metadata(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/runs/crons",
            json={
                "assistant_id": "asst-001",
                "schedule": "0 * * * *",
                "metadata": {"env": "prod"},
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /threads/{thread_id}/runs/crons  →  Run
# ---------------------------------------------------------------------------


class TestCreateCronForThread:
    """Test POST /threads/{thread_id}/runs/crons."""

    def test_creates_cron_for_thread(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/threads/thread-001/runs/crons",
            json={
                "assistant_id": "asst-001",
                "schedule": "*/10 * * * *",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        # Verify thread_id kwarg was passed
        call_kwargs = mock_cron_service.create_cron.call_args
        assert call_kwargs.kwargs.get("thread_id") == "thread-001"


# ---------------------------------------------------------------------------
# PATCH /runs/crons/{cron_id}  →  CronResponse
# ---------------------------------------------------------------------------


class TestUpdateCron:
    """Test PATCH /runs/crons/{cron_id}."""

    def test_updates_cron(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.update_cron.return_value = _cron_response(schedule="*/10 * * * *")

        resp = client.patch(
            "/runs/crons/cron-001",
            json={"schedule": "*/10 * * * *"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["cron_id"] == "cron-001"
        assert data["schedule"] == "*/10 * * * *"

    def test_returns_404_when_not_found(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.update_cron.side_effect = HTTPException(404, "Cron 'missing' not found")

        resp = client.patch(
            "/runs/crons/missing",
            json={"enabled": False},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /runs/crons/{cron_id}  →  204
# ---------------------------------------------------------------------------


class TestDeleteCron:
    """Test DELETE /runs/crons/{cron_id}."""

    def test_deletes_cron(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.delete_cron.return_value = None

        resp = client.delete("/runs/crons/cron-001")
        assert resp.status_code == 204

    def test_returns_404_when_not_found(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.delete_cron.side_effect = HTTPException(404, "Cron 'missing' not found")

        resp = client.delete("/runs/crons/missing")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /runs/crons/search  →  list[CronResponse]
# ---------------------------------------------------------------------------


class TestSearchCrons:
    """Test POST /runs/crons/search."""

    def test_returns_list(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = [
            _cron_response(cron_id="c1"),
            _cron_response(cron_id="c2"),
        ]

        resp = client.post("/runs/crons/search", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["cron_id"] == "c1"
        assert data[1]["cron_id"] == "c2"

    def test_returns_empty_list(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = []

        resp = client.post("/runs/crons/search", json={})

        assert resp.status_code == 200
        assert resp.json() == []

    def test_filters_by_assistant_id(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = []

        resp = client.post(
            "/runs/crons/search",
            json={"assistant_id": "asst-001"},
        )
        assert resp.status_code == 200
        mock_cron_service.search_crons.assert_called_once()


# ---------------------------------------------------------------------------
# POST /runs/crons/count  →  int
# ---------------------------------------------------------------------------


class TestCountCrons:
    """Test POST /runs/crons/count."""

    def test_returns_count(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.count_crons.return_value = 7

        resp = client.post("/runs/crons/count", json={})

        assert resp.status_code == 200
        assert resp.json() == 7

    def test_returns_zero(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.count_crons.return_value = 0

        resp = client.post("/runs/crons/count", json={})

        assert resp.status_code == 200
        assert resp.json() == 0

    def test_filters_by_assistant_id(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.count_crons.return_value = 3

        resp = client.post("/runs/crons/count", json={"assistant_id": "asst-001"})
        assert resp.status_code == 200
        assert resp.json() == 3

    def test_filters_by_thread_id(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.count_crons.return_value = 1

        resp = client.post("/runs/crons/count", json={"thread_id": "t-001"})
        assert resp.status_code == 200
        assert resp.json() == 1


# ---------------------------------------------------------------------------
# POST /runs/crons — extended
# ---------------------------------------------------------------------------


class TestCreateCronExtended:
    """Extended tests for POST /runs/crons."""

    def test_with_all_fields(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/runs/crons",
            json={
                "assistant_id": "asst-001",
                "schedule": "*/5 * * * *",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "metadata": {"env": "prod"},
                "config": {"temperature": 0.5},
                "webhook": "https://hooks.example.com/cb",
                "on_run_completed": "create_new",
                "enabled": False,
                "multitask_strategy": "reject",
                "stream_mode": "values",
                "stream_subgraphs": True,
            },
        )
        assert resp.status_code == 200
        assert "run_id" in resp.json()

    def test_missing_assistant_id_returns_422(self, client, mock_cron_service: AsyncMock) -> None:
        resp = client.post(
            "/runs/crons",
            json={"schedule": "*/5 * * * *"},
        )
        assert resp.status_code == 422

    def test_missing_schedule_returns_422(self, client, mock_cron_service: AsyncMock) -> None:
        resp = client.post(
            "/runs/crons",
            json={"assistant_id": "asst-001"},
        )
        assert resp.status_code == 422

    def test_service_422_propagates(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.side_effect = HTTPException(422, "Invalid cron schedule: bad")

        resp = client.post(
            "/runs/crons",
            json={"assistant_id": "asst-001", "schedule": "bad"},
        )
        assert resp.status_code == 422

    def test_service_404_propagates(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.side_effect = HTTPException(404, "Assistant 'missing' not found")

        resp = client.post(
            "/runs/crons",
            json={"assistant_id": "missing", "schedule": "*/5 * * * *"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /threads/{thread_id}/runs/crons — extended
# ---------------------------------------------------------------------------


class TestCreateCronForThreadExtended:
    """Extended tests for POST /threads/{thread_id}/runs/crons."""

    def test_with_metadata(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/threads/t-001/runs/crons",
            json={
                "assistant_id": "asst-001",
                "schedule": "0 * * * *",
                "metadata": {"team": "backend"},
            },
        )
        assert resp.status_code == 200

    def test_missing_body_returns_422(self, client, mock_cron_service: AsyncMock) -> None:
        resp = client.post("/threads/t-001/runs/crons", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# PATCH /runs/crons/{cron_id} — extended
# ---------------------------------------------------------------------------


class TestUpdateCronExtended:
    """Extended tests for PATCH /runs/crons/{cron_id}."""

    def test_updates_multiple_fields(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.update_cron.return_value = _cron_response(
            schedule="0 12 * * *",
            enabled=False,
            on_run_completed="create_new",
        )

        resp = client.patch(
            "/runs/crons/cron-001",
            json={
                "schedule": "0 12 * * *",
                "enabled": False,
                "on_run_completed": "create_new",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["schedule"] == "0 12 * * *"
        assert data["enabled"] is False
        assert data["on_run_completed"] == "create_new"

    def test_empty_update_body(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.update_cron.return_value = _cron_response()

        resp = client.patch("/runs/crons/cron-001", json={})
        assert resp.status_code == 200

    def test_update_metadata(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.update_cron.return_value = _cron_response(metadata={"new_key": "new_val"})

        resp = client.patch(
            "/runs/crons/cron-001",
            json={"metadata": {"new_key": "new_val"}},
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"] == {"new_key": "new_val"}


# ---------------------------------------------------------------------------
# DELETE /runs/crons/{cron_id} — extended
# ---------------------------------------------------------------------------


class TestDeleteCronExtended:
    """Extended tests for DELETE /runs/crons/{cron_id}."""

    def test_response_has_no_body(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.delete_cron.return_value = None

        resp = client.delete("/runs/crons/cron-001")
        assert resp.status_code == 204
        assert resp.content == b""


# ---------------------------------------------------------------------------
# POST /runs/crons/search — extended
# ---------------------------------------------------------------------------


class TestSearchCronsExtended:
    """Extended tests for POST /runs/crons/search."""

    def test_filters_by_thread_id(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = [
            _cron_response(thread_id="t-1"),
        ]

        resp = client.post("/runs/crons/search", json={"thread_id": "t-1"})
        assert resp.status_code == 200
        assert resp.json()[0]["thread_id"] == "t-1"

    def test_filters_by_enabled(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = [
            _cron_response(enabled=False),
        ]

        resp = client.post("/runs/crons/search", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()[0]["enabled"] is False

    def test_pagination_params(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = []

        resp = client.post(
            "/runs/crons/search",
            json={"limit": 5, "offset": 10},
        )
        assert resp.status_code == 200

    def test_sort_params(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = []

        resp = client.post(
            "/runs/crons/search",
            json={"sort_by": "next_run_date", "sort_order": "desc"},
        )
        assert resp.status_code == 200

    def test_response_includes_all_fields(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.search_crons.return_value = [
            _cron_response(
                cron_id="c-full",
                thread_id="t-1",
                on_run_completed="create_new",
                payload={"input": {"k": "v"}},
            ),
        ]

        resp = client.post("/runs/crons/search", json={})
        assert resp.status_code == 200
        data = resp.json()[0]
        assert data["cron_id"] == "c-full"
        assert data["thread_id"] == "t-1"
        assert data["on_run_completed"] == "create_new"
        assert data["payload"] == {"input": {"k": "v"}}
        assert data["schedule"] == "*/5 * * * *"
        assert "created_at" in data
        assert "updated_at" in data
        assert "next_run_date" in data
        assert "user_id" in data
        assert "enabled" in data


# ---------------------------------------------------------------------------
# timezone field — integration coverage
# ---------------------------------------------------------------------------


class TestTimezoneField:
    """Integration tests for the timezone field on create/update endpoints."""

    def test_create_accepts_timezone(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/runs/crons",
            json={
                "assistant_id": "asst-001",
                "schedule": "0 9 * * *",
                "timezone": "America/New_York",
            },
        )

        assert resp.status_code == 200
        call_args = mock_cron_service.create_cron.call_args
        request_obj = call_args.args[0]
        assert request_obj.timezone == "America/New_York"

    def test_create_for_thread_accepts_timezone(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/threads/t-001/runs/crons",
            json={
                "assistant_id": "asst-001",
                "schedule": "0 9 * * *",
                "timezone": "Europe/London",
            },
        )

        assert resp.status_code == 200
        call_args = mock_cron_service.create_cron.call_args
        request_obj = call_args.args[0]
        assert request_obj.timezone == "Europe/London"

    def test_update_accepts_timezone(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.update_cron.return_value = _cron_response(payload={"timezone": "Asia/Tokyo"})

        resp = client.patch(
            "/runs/crons/cron-001",
            json={"timezone": "Asia/Tokyo"},
        )

        assert resp.status_code == 200
        call_args = mock_cron_service.update_cron.call_args
        request_obj = call_args.args[1]
        assert request_obj.timezone == "Asia/Tokyo"

    def test_create_without_timezone_is_valid(self, client, mock_cron_service: AsyncMock) -> None:
        mock_cron_service.create_cron.return_value = AsyncMock()

        resp = client.post(
            "/runs/crons",
            json={"assistant_id": "asst-001", "schedule": "*/5 * * * *"},
        )

        assert resp.status_code == 200
        call_args = mock_cron_service.create_cron.call_args
        request_obj = call_args.args[0]
        assert request_obj.timezone is None
