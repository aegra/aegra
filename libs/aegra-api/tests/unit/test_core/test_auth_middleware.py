"""Unit tests for auth middleware"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from aegra_api.config import AuthConfig
from aegra_api.core import auth_middleware as auth_middleware_module
from aegra_api.core.auth_middleware import (
    LangGraphAuthBackend,
    LangGraphUser,
    get_auth_backend,
    get_auth_backend_async,
    get_auth_instance,
    on_auth_error,
)

_COUNTING_AUTH_SOURCE = """\
from langgraph_sdk import Auth

auth = Auth()


@auth.authenticate
async def authenticate(headers: dict) -> dict:
    return {
        "identity": "cached-user",
        "display_name": "Cached User",
        "is_authenticated": True,
        "permissions": ["read"],
    }
"""


def _install_auth_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a custom auth module + aegra.json and chdir so the loader finds them."""
    auth_file = tmp_path / "counting_auth.py"
    auth_file.write_text(_COUNTING_AUTH_SOURCE)
    (tmp_path / "aegra.json").write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "auth": {"path": "./counting_auth.py:auth"},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return auth_file


def _count_spec_from_file_location(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch importlib so each auth-file spec creation is recorded."""
    real_spec_from_file_location = auth_middleware_module.importlib.util.spec_from_file_location
    locations: list[str] = []

    def counting_spec(
        name: str,
        location: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> ModuleSpec | None:
        if location is not None:
            locations.append(location)
        return real_spec_from_file_location(name, location, *args, **kwargs)

    monkeypatch.setattr(
        auth_middleware_module.importlib.util,
        "spec_from_file_location",
        counting_spec,
    )
    return locations


def _block_first_auth_file_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], threading.Event, threading.Event]:
    """Hold the first auth-file spec creation until the test releases it."""
    real_spec_from_file_location = auth_middleware_module.importlib.util.spec_from_file_location
    locations: list[str] = []
    load_started = threading.Event()
    load_release = threading.Event()

    def counting_spec(
        name: str,
        location: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> ModuleSpec | None:
        if location is not None:
            locations.append(location)
        load_started.set()
        if not load_release.wait(timeout=5):
            raise TimeoutError("in-flight auth import was not released")
        return real_spec_from_file_location(name, location, *args, **kwargs)

    monkeypatch.setattr(
        auth_middleware_module.importlib.util,
        "spec_from_file_location",
        counting_spec,
    )
    return locations, load_started, load_release


def _notify_when_auth_backend_claimed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    waiters: int,
) -> threading.Event:
    """Set an event after `waiters` callers have claimed the shared fill Future."""
    claimed = threading.Event()
    guard = threading.Lock()
    count = 0
    real_claim = auth_middleware_module._claim_auth_backend_fill

    def counting_claim() -> tuple[Future[AuthenticationBackend], bool]:
        nonlocal count
        result = real_claim()
        with guard:
            count += 1
            if count >= waiters:
                claimed.set()
        return result

    monkeypatch.setattr(auth_middleware_module, "_claim_auth_backend_fill", counting_claim)
    return claimed


def _run_concurrent_calls_during_in_flight_import(
    *,
    load_started: threading.Event,
    load_release: threading.Event,
    target: Callable[[], None],
) -> None:
    """Release both callers together, then keep the first import in-flight."""
    ready = threading.Barrier(2)

    def gated() -> None:
        ready.wait()
        target()

    first = threading.Thread(target=gated)
    second = threading.Thread(target=gated)
    first.start()
    second.start()
    try:
        assert load_started.wait(timeout=5)
    finally:
        load_release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()


class TestLangGraphUser:
    """Test LangGraphUser class"""

    def test_user_initialization(self):
        """Test user initialization with user data"""
        user_data = {
            "identity": "user-123",
            "display_name": "Test User",
            "is_authenticated": True,
            "email": "test@example.com",
        }

        user = LangGraphUser(user_data)

        assert user.identity == "user-123"
        assert user.display_name == "Test User"
        assert user.is_authenticated is True

    def test_user_identity_property(self):
        """Test identity property"""
        user_data = {"identity": "test-identity"}
        user = LangGraphUser(user_data)

        assert user.identity == "test-identity"

    def test_user_display_name_default(self):
        """Test display_name defaults to identity"""
        user_data = {"identity": "test-identity"}
        user = LangGraphUser(user_data)

        assert user.display_name == "test-identity"

    def test_user_display_name_custom(self):
        """Test custom display_name"""
        user_data = {"identity": "test-identity", "display_name": "Custom Name"}
        user = LangGraphUser(user_data)

        assert user.display_name == "Custom Name"

    def test_user_is_authenticated_default(self):
        """Test is_authenticated defaults to True"""
        user_data = {"identity": "test-identity"}
        user = LangGraphUser(user_data)

        assert user.is_authenticated is True

    def test_user_is_authenticated_custom(self):
        """Test custom is_authenticated value"""
        user_data = {"identity": "test-identity", "is_authenticated": False}
        user = LangGraphUser(user_data)

        assert user.is_authenticated is False

    def test_user_getattr_existing_field(self):
        """Test __getattr__ with existing field"""
        user_data = {"identity": "test-identity", "email": "test@example.com"}
        user = LangGraphUser(user_data)

        assert user.email == "test@example.com"

    def test_user_getattr_nonexistent_field(self):
        """Test __getattr__ with non-existent field"""
        user_data = {"identity": "test-identity"}
        user = LangGraphUser(user_data)

        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            _ = user.nonexistent

    def test_user_to_dict(self):
        """Test to_dict method"""
        user_data = {
            "identity": "test-identity",
            "display_name": "Test User",
            "email": "test@example.com",
        }
        user = LangGraphUser(user_data)

        result = user.to_dict()

        assert result == user_data
        assert result is not user_data  # Should be a copy


class TestLangGraphAuthBackend:
    """Test LangGraphAuthBackend class"""

    def test_backend_initialization(self):
        """Test backend initialization"""
        with patch.object(LangGraphAuthBackend, "_load_auth_instance", return_value=None):
            backend = LangGraphAuthBackend()
            assert backend.auth_instance is None

    def test_no_auth_warning_emitted_once_at_init(self, monkeypatch: pytest.MonkeyPatch):
        """The no-auth warning fires once at startup, not on every request."""
        from aegra_api.core import auth_middleware

        warnings: list[str] = []
        monkeypatch.setattr(auth_middleware.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

        with patch.object(LangGraphAuthBackend, "_load_auth_instance", return_value=None):
            LangGraphAuthBackend()

        assert sum("single 'anonymous' identity" in w for w in warnings) == 1

    @pytest.mark.asyncio
    async def test_authenticate_does_not_warn_per_request(self, monkeypatch: pytest.MonkeyPatch):
        """authenticate() must not emit the no-auth warning; that would flood logs."""
        from aegra_api.core import auth_middleware

        with patch.object(LangGraphAuthBackend, "_load_auth_instance", return_value=None):
            backend = LangGraphAuthBackend()

        warnings: list[str] = []
        monkeypatch.setattr(auth_middleware.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {}
        for _ in range(3):
            await backend.authenticate(mock_conn)

        assert not any("single 'anonymous' identity" in w for w in warnings)

    def test_load_auth_instance_success(self):
        """Test successful auth instance loading"""
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock()

        with patch.object(LangGraphAuthBackend, "_load_auth_instance", return_value=mock_auth_instance):
            backend = LangGraphAuthBackend()

            assert backend.auth_instance == mock_auth_instance

    def test_load_auth_instance_file_not_found(self):
        """Test auth instance loading when file doesn't exist - returns None, noop handled in authenticate()"""
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.cwd", return_value=Path("/test")),
            patch("aegra_api.core.auth_middleware.load_auth_config", return_value=None),
        ):
            backend = LangGraphAuthBackend()

            # Should return None (noop handled directly in authenticate method)
            assert backend.auth_instance is None

    def test_load_auth_instance_spec_failure(self):
        """Test auth instance loading when spec creation fails"""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.cwd", return_value=Path("/test")),
            patch("importlib.util.spec_from_file_location", return_value=None),
        ):
            backend = LangGraphAuthBackend()

            assert backend.auth_instance is None

    def test_load_auth_instance_no_auth_attribute(self):
        """Test auth instance loading when module has no auth attribute"""
        mock_module = Mock()
        mock_module.auth = None

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.cwd", return_value=Path("/test")),
            patch("importlib.util.spec_from_file_location") as mock_spec,
            patch("importlib.util.module_from_spec", return_value=mock_module),
            patch("sys.modules", {}),
        ):
            mock_spec.return_value = Mock()
            mock_spec.return_value.loader = Mock()

            backend = LangGraphAuthBackend()

            assert backend.auth_instance is None

    def test_load_auth_instance_invalid_auth_type(self):
        """Test auth instance loading when auth is not Auth instance"""
        mock_module = Mock()
        mock_module.auth = "not_an_auth_instance"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.cwd", return_value=Path("/test")),
            patch("importlib.util.spec_from_file_location") as mock_spec,
            patch("importlib.util.module_from_spec", return_value=mock_module),
            patch("sys.modules", {}),
        ):
            mock_spec.return_value = Mock()
            mock_spec.return_value.loader = Mock()

            backend = LangGraphAuthBackend()

            assert backend.auth_instance is None

    def test_load_auth_instance_exception(self):
        """Test auth instance loading when exception occurs - returns None, noop handled in authenticate()"""
        # Test that exceptions during config loading are handled gracefully
        # noop is handled directly in authenticate method, not here
        with (
            patch(
                "aegra_api.core.auth_middleware.load_auth_config",
                side_effect=Exception("Test config error"),
            ),
            patch("pathlib.Path.exists", return_value=False),  # Fallback also finds nothing
            patch("pathlib.Path.cwd", return_value=Path("/test")),
        ):
            backend = LangGraphAuthBackend()

            # Should return None (noop handled directly in authenticate method)
            assert backend.auth_instance is None

    @pytest.mark.asyncio
    async def test_authenticate_noop_when_no_auth_instance(self):
        """Test that noop authentication works when no auth instance is configured"""
        backend = LangGraphAuthBackend()
        backend.auth_instance = None

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {}

        with patch("aegra_api.core.auth_middleware.settings") as mock_settings:
            mock_settings.app.AUTH_TYPE = "noop"
            result = await backend.authenticate(mock_conn)

            assert result is not None
            credentials, user = result
            assert user.identity == "anonymous"
            assert user.display_name == "Anonymous User"
            assert user.is_authenticated is True
            assert isinstance(credentials, AuthCredentials)

    @pytest.mark.asyncio
    async def test_authenticate_no_auth_instance_defaults_to_noop(self):
        """Test that no auth instance defaults to noop (anonymous) regardless of AUTH_TYPE"""
        backend = LangGraphAuthBackend()
        backend.auth_instance = None

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {}

        # Test with AUTH_TYPE=custom - should still default to noop
        with patch("aegra_api.core.auth_middleware.settings") as mock_settings:
            mock_settings.app.AUTH_TYPE = "custom"
            result = await backend.authenticate(mock_conn)

            # Should return anonymous user even when AUTH_TYPE=custom
            assert result is not None
            credentials, user = result
            assert user.identity == "anonymous"
            assert user.display_name == "Anonymous User"

    @pytest.mark.asyncio
    async def test_authenticate_no_handler(self):
        """Test authentication when no handler is configured"""
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = None

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        mock_conn = Mock(spec=HTTPConnection)

        result = await backend.authenticate(mock_conn)

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_success(self):
        """Test successful authentication"""
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock(
            return_value={
                "identity": "user-123",
                "display_name": "Test User",
                "permissions": ["read", "write"],
            }
        )

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {
            "authorization": b"Bearer token123",
            "content-type": b"application/json",
        }

        credentials, user = await backend.authenticate(mock_conn)

        assert isinstance(credentials, AuthCredentials)
        assert credentials.scopes == ["read", "write"]
        assert isinstance(user, LangGraphUser)
        assert user.identity == "user-123"
        assert user.display_name == "Test User"

    @pytest.mark.asyncio
    async def test_authenticate_success_string_permissions(self):
        """Test authentication with string permissions"""
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock(
            return_value={"identity": "user-123", "permissions": "admin"}
        )

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": b"Bearer token123"}

        credentials, user = await backend.authenticate(mock_conn)

        assert credentials.scopes == ["admin"]

    @pytest.mark.asyncio
    async def test_authenticate_invalid_user_data(self):
        """Test authentication with invalid user data"""
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock(return_value=None)

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": b"Bearer token123"}

        with pytest.raises(AuthenticationError, match="Authentication system error"):
            await backend.authenticate(mock_conn)

    @pytest.mark.asyncio
    async def test_authenticate_missing_identity(self):
        """Test authentication with missing identity field"""
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock(return_value={"display_name": "Test User"})

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": b"Bearer token123"}

        with pytest.raises(AuthenticationError, match="Authentication system error"):
            await backend.authenticate(mock_conn)

    @pytest.mark.asyncio
    async def test_authenticate_http_exception(self):
        """Test authentication with HTTP exception"""
        mock_auth_instance = Mock()

        # Create a mock exception with detail attribute
        mock_http_exception = Exception("Auth failed")
        mock_http_exception.detail = "Invalid token"
        mock_auth_instance._authenticate_handler = AsyncMock(side_effect=mock_http_exception)

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        # Mock the Auth.exceptions.HTTPException to be the same as our exception
        with patch("aegra_api.core.auth_middleware.Auth") as mock_auth:
            mock_auth.exceptions.HTTPException = Exception

            mock_conn = Mock(spec=HTTPConnection)
            mock_conn.headers = {"authorization": b"Bearer token123"}

            with pytest.raises(AuthenticationError, match="Invalid token"):
                await backend.authenticate(mock_conn)

    @pytest.mark.asyncio
    async def test_authenticate_headers_conversion(self):
        """Test header conversion for different types"""
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock(return_value={"identity": "user-123"})

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {
            b"authorization": b"Bearer token123",  # bytes key and value
            "content-type": "application/json",  # str key and value
            b"user-agent": "test-agent",  # bytes key, str value
        }

        await backend.authenticate(mock_conn)

        # Verify headers were converted properly
        expected_headers = {
            "authorization": "Bearer token123",
            "content-type": "application/json",
            "user-agent": "test-agent",
        }
        mock_auth_instance._authenticate_handler.assert_called_once_with(expected_headers)


class TestGetAuthBackend:
    """Test get_auth_backend function"""

    def test_get_auth_backend_noop(self) -> None:
        """Test getting auth backend with noop type"""
        with patch.dict(os.environ, {"AUTH_TYPE": "noop"}):
            backend = get_auth_backend()
            assert isinstance(backend, LangGraphAuthBackend)

    def test_get_auth_backend_custom(self) -> None:
        """Test getting auth backend with custom type"""
        with patch.dict(os.environ, {"AUTH_TYPE": "custom"}):
            backend = get_auth_backend()
            assert isinstance(backend, LangGraphAuthBackend)

    def test_get_auth_backend_unknown(self) -> None:
        """Test getting auth backend with unknown type"""
        with patch.dict(os.environ, {"AUTH_TYPE": "unknown"}):
            backend = get_auth_backend()
            assert isinstance(backend, LangGraphAuthBackend)

    def test_get_auth_backend_default(self) -> None:
        """Test getting auth backend with no AUTH_TYPE set"""
        with patch.dict(os.environ, {}, clear=True):
            backend = get_auth_backend()
            assert isinstance(backend, LangGraphAuthBackend)

    def test_get_auth_backend_returns_same_instance(self) -> None:
        """get_auth_backend is a process singleton; repeated calls must not rebuild it."""
        first = get_auth_backend()
        second = get_auth_backend()
        assert first is second


class TestAuthModuleLoadOnce:
    """Custom auth modules must be imported once per config path, not per request."""

    def test_get_auth_backend_loads_auth_file_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated get_auth_backend() calls import the auth file a single time."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations = _count_spec_from_file_location(monkeypatch)

        backends = [get_auth_backend() for _ in range(5)]

        assert len({id(backend) for backend in backends}) == 1
        assert isinstance(backends[0], LangGraphAuthBackend)
        assert backends[0].auth_instance is not None
        assert locations == [str(auth_file.resolve())]

    def test_langgraph_auth_backend_init_reuses_cached_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct backend construction must still reuse the path-keyed module cache."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations = _count_spec_from_file_location(monkeypatch)

        first = LangGraphAuthBackend()
        second = LangGraphAuthBackend()
        via_factory = get_auth_backend()

        assert isinstance(via_factory, LangGraphAuthBackend)
        assert first.auth_instance is not None
        assert first.auth_instance is second.auth_instance
        assert first.auth_instance is via_factory.auth_instance
        assert get_auth_instance() is first.auth_instance
        assert locations == [str(auth_file.resolve())]

    def test_concurrent_first_misses_load_auth_module_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two threads racing an empty cache must import/construct Auth once."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations, load_started, load_release = _block_first_auth_file_spec(monkeypatch)
        backends: list[LangGraphAuthBackend] = []
        errors: list[Exception] = []

        def construct_backend() -> None:
            try:
                backends.append(LangGraphAuthBackend())
            except Exception as exc:
                errors.append(exc)

        _run_concurrent_calls_during_in_flight_import(
            load_started=load_started,
            load_release=load_release,
            target=construct_backend,
        )

        assert errors == []
        assert len(backends) == 2
        assert backends[0].auth_instance is not None
        assert backends[0].auth_instance is backends[1].auth_instance
        assert locations == [str(auth_file.resolve())]

    def test_concurrent_get_auth_backend_constructs_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two threads racing get_auth_backend() must share one backend and one import."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations, load_started, load_release = _block_first_auth_file_spec(monkeypatch)
        backends: list[object] = []
        errors: list[Exception] = []

        def call_factory() -> None:
            try:
                backends.append(get_auth_backend())
            except Exception as exc:
                errors.append(exc)

        _run_concurrent_calls_during_in_flight_import(
            load_started=load_started,
            load_release=load_release,
            target=call_factory,
        )

        assert errors == []
        assert len(backends) == 2
        assert backends[0] is backends[1]
        assert isinstance(backends[0], LangGraphAuthBackend)
        assert backends[0].auth_instance is not None
        assert locations == [str(auth_file.resolve())]

    @pytest.mark.asyncio
    async def test_async_fill_does_not_run_on_event_loop_thread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Async init must complete the shared fill off the event-loop thread."""
        _install_auth_config(tmp_path, monkeypatch)
        loop_thread_id = threading.get_ident()
        fill_thread_ids: list[int] = []
        real_complete = auth_middleware_module._complete_auth_backend_fill

        def tracking_complete(fut: Future[AuthenticationBackend]) -> AuthenticationBackend:
            fill_thread_ids.append(threading.get_ident())
            return real_complete(fut)

        monkeypatch.setattr(auth_middleware_module, "_complete_auth_backend_fill", tracking_complete)

        backend = await get_auth_backend_async()

        assert isinstance(backend, LangGraphAuthBackend)
        assert backend.auth_instance is not None
        assert fill_thread_ids
        assert loop_thread_id not in fill_thread_ids

    @pytest.mark.asyncio
    async def test_event_loop_sync_waiter_does_not_deadlock_async_fill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sync waiter on this loop must not block the worker that owns the fill."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations, load_started, load_release = _block_first_auth_file_spec(monkeypatch)
        loop_thread_id = threading.get_ident()
        loop_waiting = threading.Event()
        errors: list[BaseException] = []
        real_future_result = Future.result

        def tracking_result(
            self: Future[AuthenticationBackend],
            timeout: float | None = None,
        ) -> AuthenticationBackend:
            if self is auth_middleware_module._auth_backend_fill and threading.get_ident() == loop_thread_id:
                loop_waiting.set()
            return real_future_result(self, timeout)

        monkeypatch.setattr(Future, "result", tracking_result)

        def release_after_loop_waits() -> None:
            if not loop_waiting.wait(timeout=5):
                errors.append(TimeoutError("event-loop waiter did not block on shared fill"))
            load_release.set()

        releaser = threading.Thread(target=release_after_loop_waits)
        releaser.start()
        try:
            async_task = asyncio.create_task(get_auth_backend_async())
            assert await asyncio.to_thread(load_started.wait, 5)
            sync_backend = get_auth_backend()
            async_backend = await asyncio.wait_for(async_task, timeout=5)
        finally:
            load_release.set()
            releaser.join(timeout=5)

        assert errors == []
        assert not releaser.is_alive()
        assert not async_task.cancelled()
        assert sync_backend is async_backend
        assert isinstance(async_backend, LangGraphAuthBackend)
        assert async_backend.auth_instance is not None
        assert locations == [str(auth_file.resolve())]

    @pytest.mark.asyncio
    async def test_concurrent_async_first_misses_load_auth_module_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two async tasks racing an empty cache must import/construct Auth once."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations = _count_spec_from_file_location(monkeypatch)
        barrier = asyncio.Barrier(2)
        in_flight = 0
        max_in_flight = 0

        async def load_backend() -> object:
            nonlocal in_flight, max_in_flight
            await barrier.wait()
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                return await get_auth_backend_async()
            finally:
                in_flight -= 1

        first, second = await asyncio.gather(load_backend(), load_backend())

        assert max_in_flight == 2
        assert first is second
        assert isinstance(first, LangGraphAuthBackend)
        assert first.auth_instance is not None
        assert locations == [str(auth_file.resolve())]

    @pytest.mark.asyncio
    async def test_sync_thread_and_async_first_miss_load_auth_module_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A thread get_auth_backend() racing get_auth_backend_async() must import once."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations, load_started, load_release = _block_first_auth_file_spec(monkeypatch)
        both_claimed = _notify_when_auth_backend_claimed(monkeypatch, waiters=2)
        sync_backends: list[object] = []
        errors: list[BaseException] = []
        sync_ready = threading.Event()
        start_fill = threading.Event()

        def call_sync() -> None:
            sync_ready.set()
            if not start_fill.wait(timeout=5):
                errors.append(TimeoutError("sync caller was not released into the empty-cache fill"))
                return
            try:
                sync_backends.append(get_auth_backend())
            except Exception as exc:
                errors.append(exc)

        sync_thread = threading.Thread(target=call_sync)
        sync_thread.start()
        try:
            assert await asyncio.to_thread(sync_ready.wait, 5)

            async def call_async() -> object:
                if not await asyncio.to_thread(start_fill.wait, 5):
                    raise TimeoutError("async caller was not released into the empty-cache fill")
                return await get_auth_backend_async()

            async_task = asyncio.create_task(call_async())
            start_fill.set()
            assert await asyncio.to_thread(load_started.wait, 5)
            assert await asyncio.to_thread(both_claimed.wait, 5)
            load_release.set()
            async_backend = await asyncio.wait_for(async_task, timeout=5)
        finally:
            load_release.set()
            sync_thread.join(timeout=5)

        assert errors == []
        assert not sync_thread.is_alive()
        assert len(sync_backends) == 1
        assert sync_backends[0] is async_backend
        assert isinstance(async_backend, LangGraphAuthBackend)
        assert async_backend.auth_instance is not None
        assert locations == [str(auth_file.resolve())]

    @pytest.mark.asyncio
    async def test_cancelled_async_joiner_does_not_cancel_shared_fill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling one async joiner must not cancel the shared fill or other waiters."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations, load_started, load_release = _block_first_auth_file_spec(monkeypatch)
        all_claimed = _notify_when_auth_backend_claimed(monkeypatch, waiters=3)
        sync_backends: list[object] = []
        errors: list[BaseException] = []

        def call_sync() -> None:
            try:
                sync_backends.append(get_auth_backend())
            except Exception as exc:
                errors.append(exc)

        sync_thread = threading.Thread(target=call_sync)
        sync_thread.start()
        try:
            assert await asyncio.to_thread(load_started.wait, 5)

            cancelled_joiner = asyncio.create_task(get_auth_backend_async())
            surviving_joiner = asyncio.create_task(get_auth_backend_async())
            assert await asyncio.to_thread(all_claimed.wait, 5)

            shared_fill = auth_middleware_module._auth_backend_fill
            assert shared_fill is not None
            cancelled_joiner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_joiner
            assert not shared_fill.cancelled()
            assert not shared_fill.done()

            load_release.set()
            surviving_backend = await asyncio.wait_for(surviving_joiner, timeout=5)
            leftover = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
            if leftover:
                _done, pending = await asyncio.wait(leftover, timeout=5)
                assert not pending
        finally:
            load_release.set()
            sync_thread.join(timeout=5)

        assert errors == []
        assert not sync_thread.is_alive()
        assert len(sync_backends) == 1
        assert sync_backends[0] is surviving_backend
        assert isinstance(surviving_backend, LangGraphAuthBackend)
        assert surviving_backend.auth_instance is not None
        assert locations == [str(auth_file.resolve())]

    @pytest.mark.asyncio
    async def test_sync_and_async_joiners_see_same_fill_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed fill must set_exception so sync and async callers see the same error."""
        both_claimed = _notify_when_auth_backend_claimed(monkeypatch, waiters=2)
        sync_errors: list[BaseException] = []
        sync_ready = threading.Event()
        start_fill = threading.Event()

        def exploding_init(self: LangGraphAuthBackend) -> None:
            if not both_claimed.wait(timeout=5):
                raise TimeoutError("second caller did not join before fill failed")
            raise RuntimeError("backend fill failed")

        monkeypatch.setattr(LangGraphAuthBackend, "__init__", exploding_init)

        def call_sync() -> None:
            sync_ready.set()
            if not start_fill.wait(timeout=5):
                sync_errors.append(TimeoutError("sync caller was not released into the empty-cache fill"))
                return
            try:
                get_auth_backend()
            except Exception as exc:
                sync_errors.append(exc)

        sync_thread = threading.Thread(target=call_sync)
        sync_thread.start()
        try:
            assert await asyncio.to_thread(sync_ready.wait, 5)

            async def call_async() -> None:
                if not await asyncio.to_thread(start_fill.wait, 5):
                    raise TimeoutError("async caller was not released into the empty-cache fill")
                await get_auth_backend_async()

            async_task = asyncio.create_task(call_async())
            start_fill.set()
            with pytest.raises(RuntimeError, match="backend fill failed") as async_raised:
                await asyncio.wait_for(async_task, timeout=5)
        finally:
            sync_thread.join(timeout=5)

        assert not sync_thread.is_alive()
        assert len(sync_errors) == 1
        assert sync_errors[0] is async_raised.value

    def test_missing_auth_file_is_not_reprobed_per_init(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing auth file is resolved once; later inits reuse that result."""
        (tmp_path / "aegra.json").write_text(
            json.dumps(
                {
                    "graphs": {"test": "./test.py:graph"},
                    "auth": {"path": "./does_not_exist.py:auth"},
                }
            )
        )
        monkeypatch.chdir(tmp_path)
        config_reads: list[int] = []
        real_load_auth_config = auth_middleware_module.load_auth_config

        def counting_load_auth_config() -> AuthConfig | None:
            config_reads.append(1)
            return real_load_auth_config()

        monkeypatch.setattr(auth_middleware_module, "load_auth_config", counting_load_auth_config)

        first = LangGraphAuthBackend()
        second = LangGraphAuthBackend()

        assert first.auth_instance is None
        assert second.auth_instance is None
        assert len(config_reads) == 1

    @pytest.mark.asyncio
    async def test_cached_auth_handler_still_authenticates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Caching the Auth object must not change authenticate() results."""
        _install_auth_config(tmp_path, monkeypatch)

        backend = get_auth_backend()
        assert isinstance(backend, LangGraphAuthBackend)

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer unused"}

        first = await backend.authenticate(mock_conn)
        second = await get_auth_backend().authenticate(mock_conn)

        assert first is not None and second is not None
        _, first_user = first
        _, second_user = second
        assert first_user.identity == "cached-user"
        assert second_user.identity == "cached-user"
        assert first_user.display_name == "Cached User"

    def test_clear_auth_loader_caches_allows_reinitialization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clearing loader caches must drop the in-flight Future so a later call reloads."""
        auth_file = _install_auth_config(tmp_path, monkeypatch)
        locations = _count_spec_from_file_location(monkeypatch)

        first = get_auth_backend()
        auth_middleware_module._clear_auth_loader_caches()
        assert auth_middleware_module._auth_backend_fill is None
        second = get_auth_backend()

        assert first is not second
        assert isinstance(second, LangGraphAuthBackend)
        assert first.auth_instance is not None
        assert second.auth_instance is not None
        assert locations == [str(auth_file.resolve()), str(auth_file.resolve())]


class TestOnAuthError:
    """Test on_auth_error function"""

    def test_on_auth_error_response(self):
        """Test auth error response format"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.url = "http://example.com/api/test"

        exc = AuthenticationError("Invalid credentials")

        response = on_auth_error(mock_conn, exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 401

        # Check response content structure
        content = response.body.decode()
        assert '"error":"unauthorized"' in content
        assert '"message":"Invalid credentials"' in content
        assert '"authentication_required":true' in content

    def test_on_auth_error_different_message(self):
        """Test auth error with different message"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.url = "http://example.com/api/test"

        exc = AuthenticationError("Token expired")

        response = on_auth_error(mock_conn, exc)

        content = response.body.decode()
        assert '"message":"Token expired"' in content

    def test_on_auth_error_empty_message(self):
        """Test auth error with empty message"""
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.url = "http://example.com/api/test"

        exc = AuthenticationError("")

        response = on_auth_error(mock_conn, exc)

        content = response.body.decode()
        assert '"message":""' in content


class TestAuthMiddlewareIntegration:
    """Test auth middleware integration scenarios"""

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        # Mock auth instance
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock(
            return_value={
                "identity": "user-123",
                "display_name": "Test User",
                "email": "test@example.com",
                "permissions": ["read", "write", "admin"],
            }
        )

        # Create backend
        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        # Mock connection
        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {
            "authorization": "Bearer valid-token",
            "content-type": "application/json",
        }

        # Authenticate
        credentials, user = await backend.authenticate(mock_conn)

        # Verify results
        assert isinstance(credentials, AuthCredentials)
        assert credentials.scopes == ["read", "write", "admin"]

        assert isinstance(user, LangGraphUser)
        assert user.identity == "user-123"
        assert user.display_name == "Test User"
        assert user.email == "test@example.com"
        assert user.is_authenticated is True

        # Test user dict conversion
        user_dict = user.to_dict()
        assert user_dict["identity"] == "user-123"
        assert user_dict["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_authentication_error_flow(self):
        """Test authentication error handling flow"""
        # Mock auth instance that raises exception
        mock_auth_instance = Mock()
        mock_auth_instance._authenticate_handler = AsyncMock(side_effect=Exception("Invalid token"))

        backend = LangGraphAuthBackend()
        backend.auth_instance = mock_auth_instance

        mock_conn = Mock(spec=HTTPConnection)
        mock_conn.headers = {"authorization": "Bearer invalid-token"}

        # Should raise AuthenticationError
        with pytest.raises(AuthenticationError):
            await backend.authenticate(mock_conn)

        # Test error response
        exc = AuthenticationError("Invalid token")
        response = on_auth_error(mock_conn, exc)

        assert response.status_code == 401
        content = response.body.decode()
        assert '"error":"unauthorized"' in content
        assert '"message":"Invalid token"' in content
