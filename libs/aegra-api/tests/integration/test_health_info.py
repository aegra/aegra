"""Integration tests for health and info endpoints."""

from aegra_api.settings import settings
from tests.fixtures.clients import create_test_app, make_client


def test_info_reports_cron_flag_from_runtime_settings(monkeypatch) -> None:
    """GET /info should expose the current CRON_ENABLED runtime flag."""
    monkeypatch.setattr(settings.cron, "CRON_ENABLED", False)

    client = make_client(create_test_app())
    response = client.get("/info")

    assert response.status_code == 200
    assert response.json()["flags"] == {"assistants": True, "crons": False}
