"""Tests for CORS middleware configuration in main.py

These tests verify that CORS configuration is properly applied regardless of
whether a custom app is defined, specifically testing the fix for the issue
where expose_headers was not applied in the non-custom-app code path.
"""

import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware


def find_cors_middleware(app: FastAPI) -> Middleware | None:
    """Find the CORSMiddleware in the app's middleware stack."""
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            return middleware
    return None


@pytest.fixture
def isolated_module_reload(tmp_path, monkeypatch):
    """Fixture that provides isolated module reloading with proper cleanup.

    This saves the current module state before the test and restores it after,
    preventing test pollution for subsequent tests.
    """
    monkeypatch.chdir(tmp_path)

    # Save original module state - make a copy of the dict
    original_modules = dict(sys.modules)

    # Save and update the settings AEGRA_CONFIG to point to our temp config file
    from aegra_api.settings import settings

    original_aegra_config = settings.app.AEGRA_CONFIG
    # Use object.__setattr__ to bypass pydantic's frozen model protection
    object.__setattr__(settings.app, "AEGRA_CONFIG", str(tmp_path / "aegra.json"))

    yield tmp_path

    # Restore original AEGRA_CONFIG
    object.__setattr__(settings.app, "AEGRA_CONFIG", original_aegra_config)

    # Restore original module state after test
    # First, remove any new modules that were added
    current_modules = list(sys.modules.keys())
    for mod in current_modules:
        if mod not in original_modules:
            del sys.modules[mod]

    # Then restore removed AND replaced modules: reload_main_module swaps in
    # fresh aegra_api.config/main objects, and leaking those breaks any later
    # test that patched globals on the originals.
    for mod, module in original_modules.items():
        if sys.modules.get(mod) is not module:
            sys.modules[mod] = module


def reload_main_module() -> types.ModuleType:
    """Clear aegra_api.main and aegra_api.config from cache and reimport main.

    Note: We only clear main and config modules. Settings module is NOT cleared
    to avoid test pollution - settings object is modified directly instead.
    """
    # Remove main, config, and the aegra_api package reference to force fresh imports
    # The order matters - remove submodules first, then package
    modules_to_remove = [
        k for k in list(sys.modules.keys()) if k in ("aegra_api.main", "aegra_api.config", "aegra_api")
    ]
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]

    # Force fresh import
    import importlib

    import aegra_api.main as main_module

    importlib.reload(main_module)

    return main_module


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_cors_includes_expose_headers(isolated_module_reload):
    """Test that default CORS config includes Content-Location and Location in expose_headers.

    This is critical for LangGraph SDK stream reconnection (reconnectOnMount) to work,
    as the SDK needs to read the Content-Location header from streaming responses.
    """
    tmp_path = isolated_module_reload

    # Create minimal config without http section
    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
            }
        )
    )

    main = reload_main_module()

    cors_middleware = find_cors_middleware(main.app)
    assert cors_middleware is not None, "CORS middleware should be present"

    # Check that expose_headers includes the required headers
    expose_headers = cors_middleware.kwargs.get("expose_headers", [])
    assert "Content-Location" in expose_headers, "Content-Location should be in expose_headers by default"
    assert "Location" in expose_headers, "Location should be in expose_headers by default"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cors_config_without_custom_app_applies_settings(isolated_module_reload):
    """Test that http.cors config is applied even without http.app defined.

    This is the main fix - previously CORS config was only applied when
    a custom app was defined.
    """
    tmp_path = isolated_module_reload

    # Create config with cors but no custom app
    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "cors": {
                        "allow_origins": ["https://myapp.example.com"],
                        "expose_headers": [
                            "Content-Location",
                            "Location",
                            "X-Request-ID",
                        ],
                        "max_age": 3600,
                    },
                },
            }
        )
    )

    main = reload_main_module()

    cors_middleware = find_cors_middleware(main.app)
    assert cors_middleware is not None, "CORS middleware should be present"

    # Verify custom settings were applied
    assert cors_middleware.kwargs.get("allow_origins") == ["https://myapp.example.com"], (
        "Custom allow_origins should be applied"
    )

    expose_headers = cors_middleware.kwargs.get("expose_headers", [])
    assert "X-Request-ID" in expose_headers, "Custom expose_headers should be applied"
    assert "Content-Location" in expose_headers
    assert "Location" in expose_headers

    assert cors_middleware.kwargs.get("max_age") == 3600, "Custom max_age should be applied"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cors_expose_headers_defaults_when_not_specified(isolated_module_reload):
    """Test that expose_headers defaults to Content-Location and Location when not specified in config."""
    tmp_path = isolated_module_reload

    # Create config with cors but without expose_headers
    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "cors": {
                        "allow_origins": ["*"],
                        # Note: expose_headers is NOT specified
                    },
                },
            }
        )
    )

    main = reload_main_module()

    cors_middleware = find_cors_middleware(main.app)
    assert cors_middleware is not None, "CORS middleware should be present"

    # Should have default expose_headers even though cors config exists but
    # doesn't specify expose_headers
    expose_headers = cors_middleware.kwargs.get("expose_headers", [])
    assert "Content-Location" in expose_headers, "Content-Location should default when not specified in cors config"
    assert "Location" in expose_headers, "Location should default when not specified in cors config"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_cors_credentials_false_with_wildcard_origins(isolated_module_reload: Path) -> None:
    """Test that allow_credentials defaults to False when origins is ["*"].

    Wildcard origins + credentials is insecure: any site could make
    credentialed requests. The safe default is credentials=False.
    """
    tmp_path = isolated_module_reload

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
            }
        )
    )

    main = reload_main_module()

    cors_middleware = find_cors_middleware(main.app)
    assert cors_middleware is not None, "CORS middleware should be present"
    assert cors_middleware.kwargs.get("allow_credentials") is False, (
        "allow_credentials should default to False with wildcard origins"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cors_credentials_true_with_concrete_origins(isolated_module_reload: Path) -> None:
    """Test that allow_credentials defaults to True when concrete origins are set."""
    tmp_path = isolated_module_reload

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "cors": {
                        "allow_origins": ["https://myapp.example.com"],
                    },
                },
            }
        )
    )

    main = reload_main_module()

    cors_middleware = find_cors_middleware(main.app)
    assert cors_middleware is not None, "CORS middleware should be present"
    assert cors_middleware.kwargs.get("allow_credentials") is True, (
        "allow_credentials should default to True with concrete origins"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cors_credentials_explicit_override(isolated_module_reload: Path) -> None:
    """Test that explicit allow_credentials setting overrides the default."""
    tmp_path = isolated_module_reload

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "cors": {
                        "allow_origins": ["*"],
                        "allow_credentials": True,
                    },
                },
            }
        )
    )

    main = reload_main_module()

    cors_middleware = find_cors_middleware(main.app)
    assert cors_middleware is not None, "CORS middleware should be present"
    assert cors_middleware.kwargs.get("allow_credentials") is True, (
        "Explicit allow_credentials=True should override the wildcard default"
    )
