"""Pytest configuration for auth E2E tests.

Skipped by default (pytest.ini -m "not auth_only").
Run with: make e2e-auth  or  pytest -m auth_only

This conftest skips the suite if the server does not have auth enabled.
Start the server with aegra.auth.json (see README.md) or docker-compose.auth.yml.
"""

import pytest

from aegra_api.settings import settings
from tests.e2e._utils import check_server_has_auth


@pytest.fixture(scope="session", autouse=True)
def skip_if_no_auth() -> None:
    """Skip auth E2E tests if the server is missing or auth is disabled."""
    server_url = settings.app.SERVER_URL

    has_auth = check_server_has_auth(server_url)

    if has_auth is False:
        pytest.skip(
            "Server is running but does not have auth enabled. "
            "Start with: make e2e-auth  or  "
            "AEGRA_CONFIG=aegra.auth.json AUTH_TYPE=custom (see README.md)"
        )
    elif has_auth is None:
        pytest.skip(
            f"Could not connect to server at {server_url} or determine auth status. "
            "Start an auth-enabled server with make e2e-auth or docker-compose.auth.yml."
        )
