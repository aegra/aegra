"""Unit test specific fixtures

Unit tests should be fast and isolated, with no external dependencies.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_observability_manager():
    """Reset the global observability manager before each test.

    This ensures tests don't interfere with each other when they reload
    modules that create and register observability providers.
    """
    from aegra_api.observability.base import get_observability_manager

    manager = get_observability_manager()
    # Clear all registered providers before each test
    manager._providers.clear()
    yield
    # Clean up after test
    manager._providers.clear()


@pytest.fixture(autouse=True)
def _disable_cron_quota_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the per-user cron quota in unit tests.

    The quota issues an extra ``COUNT(*)`` against the database which forces
    every CronService unit test to mock another scalar() call. Quota
    behaviour is exercised in dedicated tests that override this fixture.

    We patch the ``settings`` reference inside ``cron_service`` directly
    because tests like ``test_main`` may ``importlib.reload`` the settings
    module, leaving ``cron_service``'s import binding pointing at a stale
    ``Settings`` instance that an ``aegra_api.settings.settings`` patch
    would not reach.
    """
    from aegra_api.services import cron_service as _cs

    monkeypatch.setattr(_cs.settings.cron, "CRON_MAX_PER_USER", 0)
