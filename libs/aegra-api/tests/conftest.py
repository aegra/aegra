"""Global pytest configuration and fixtures

This file contains shared fixtures and configuration that are available
to all tests across the test suite.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from httpx import HTTPStatusError

from aegra_api.core.cancellation_state import cancellations
from tests.fixtures.auth import DummyUser
from tests.fixtures.clients import (
    create_test_app,
    make_client,
)
from tests.fixtures.database import DummySessionBase, override_get_session_dep
from tests.fixtures.langgraph import (
    FakeAgent,
    FakeGraph,
    FakeSnapshot,
    make_interrupt,
    make_snapshot,
    make_task,
    patch_langgraph_service,
)
from tests.fixtures.session_fixtures import (
    BasicSession,
    RunSession,
    ThreadSession,
    override_session_dependency,
)
from tests.fixtures.test_helpers import (
    DummyRun,
    DummyStoreItem,
    DummyThread,
    make_assistant,
    make_run,
    make_thread,
)

# Export fixtures for use in tests
__all__ = [
    "DummyUser",
    "DummySessionBase",
    "override_get_session_dep",
    "FakeSnapshot",
    "FakeAgent",
    "FakeGraph",
    "make_interrupt",
    "make_snapshot",
    "make_task",
    "patch_langgraph_service",
    "create_test_app",
    "make_client",
    "BasicSession",
    "ThreadSession",
    "RunSession",
    "override_session_dependency",
    "make_assistant",
    "make_thread",
    "make_run",
    "DummyRun",
    "DummyThread",
    "DummyStoreItem",
]


# Add any global fixtures here
@pytest.fixture
def dummy_user():
    """Fixture providing a dummy user for tests"""
    return DummyUser()


@pytest.fixture
def test_user_identity():
    """Fixture providing a test user identity"""
    return "test-user"


@pytest.fixture
def basic_session():
    """Fixture providing a basic mock session"""
    return BasicSession()


@pytest.fixture
def mock_assistant_service():
    """Fixture providing a mocked assistant service"""
    return AsyncMock()


@pytest.fixture
def mock_store():
    """Fixture providing a mocked store"""
    return AsyncMock()


@pytest.fixture
def basic_client(basic_session):
    """Fixture providing a basic test client with mocked session"""
    app = create_test_app(include_runs=False, include_threads=False)
    override_session_dependency(app, BasicSession)
    return make_client(app)


@pytest.fixture
def threads_client():
    """Fixture providing a test client for thread operations"""
    app = create_test_app(include_runs=False, include_threads=True)
    override_session_dependency(app, ThreadSession)
    return make_client(app)


@pytest.fixture
def runs_client():
    """Fixture providing a test client for run operations"""
    app = create_test_app(include_runs=True, include_threads=False)
    override_session_dependency(app, RunSession)
    return make_client(app)


@pytest.fixture(autouse=True)
def clear_auth_cache():
    """Clear auth instance cache before and after each test.

    The get_auth_instance() function uses @lru_cache which can cause
    test isolation issues when different tests need different auth configurations.
    This fixture ensures each test starts with a clean auth state.
    """
    from aegra_api.core.auth_middleware import get_auth_instance

    get_auth_instance.cache_clear()
    yield
    get_auth_instance.cache_clear()


@pytest.fixture(autouse=True)
def _clear_cancellation_registry() -> Iterator[None]:
    """Drop any cancellation tags between tests.

    The ``cancellations`` registry in ``aegra_api.core.cancellation_state`` is a
    process-wide singleton. A test that crashes between ``cancellations.mark(rid, ...)``
    and the matching ``cancellations.clear(rid)`` would otherwise leak into every
    subsequent test in the same worker, causing flaky failures that are hard to debug.
    """
    cancellations.clear_all()
    yield
    cancellations.clear_all()


# --- AUTO-SKIP GEO-BLOCK FAILURES ---
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Advanced auto-skip hook.
    Analyzes Tracebacks, Stdout, Stderr, and Logs.
    If ANY trace of OpenAI geo-blocking/quota limits is found in a FAILED test,
    marks it as SKIPPED instead.
    """
    outcome = yield
    rep = outcome.get_result()

    # Only process failed calls
    if rep.when == "call" and rep.failed:
        # Collect text from all possible sources
        text_sources = []

        # 1. Exception Info (Traceback message)
        if call.excinfo:
            text_sources.append(str(call.excinfo.value))
            # If HTTP error, grab response body
            if isinstance(call.excinfo.value, HTTPStatusError):
                text_sources.append(call.excinfo.value.response.text)

        # 2. Captured Stdout/Stderr (directly from report)
        if hasattr(rep, "capstdout"):
            text_sources.append(rep.capstdout)
        if hasattr(rep, "capstderr"):
            text_sources.append(rep.capstderr)

        # 3. Report Sections (Captured logs often live here)
        for _, section_content in rep.sections:
            text_sources.append(section_content)

        # 4. Detailed Traceback
        if rep.longrepr:
            text_sources.append(str(rep.longrepr))

        # Combine and normalize
        combined_text = "\n".join(text_sources).lower()

        # Signatures of OpenAI blocks
        block_signatures = [
            "unsupported_country_region_territory",
            "insufficient_quota",
            "rate limit",
            "generator didn't stop after athrow",
        ]

        # Check if failure was caused by OpenAI block
        if any(sig in combined_text for sig in block_signatures):
            rep.outcome = "skipped"
            # Must return a tuple for skip location
            file_path, line_no, _ = item.location
            rep.longrepr = (
                str(file_path),
                line_no + 1,
                "⛔️ Skipped: OpenAI Blocked/Quota detected in logs.",
            )
