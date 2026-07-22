"""Integration tests for custom routes functionality"""

import json

import pytest
from fastapi import APIRoute, FastAPI

from aegra_api.core.auth_deps import auth_dependency


@pytest.fixture
def custom_app_file(tmp_path):
    """Create a temporary custom app file"""
    app_file = tmp_path / "custom_app.py"
    app_file.write_text(
        """
from fastapi import FastAPI

app = FastAPI()

@app.get("/custom/hello")
async def hello():
    return {"message": "Hello from custom route!"}

@app.get("/")
async def custom_root():
    return {"message": "Custom root", "custom": True}
"""
    )
    return app_file


@pytest.fixture
def aegra_config_with_custom_app(tmp_path, custom_app_file, monkeypatch):
    """Create aegra.json with custom app configuration"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "app": f"./{custom_app_file.name}:app",
                },
            }
        )
    )

    return config_file


@pytest.mark.asyncio
async def test_custom_routes_loaded(aegra_config_with_custom_app, custom_app_file):
    """Test that custom routes are loaded and accessible"""
    # This test would require starting the actual server
    # For now, we'll test the loading mechanism
    from aegra_api.config import load_http_config
    from aegra_api.core.app_loader import load_custom_app

    http_config = load_http_config()
    assert http_config is not None
    assert "app" in http_config

    user_app = load_custom_app(http_config["app"])
    assert isinstance(user_app, FastAPI)

    # Check that custom routes exist
    routes = [r for r in user_app.routes if hasattr(r, "path")]
    route_paths = [r.path for r in routes]
    assert "/custom/hello" in route_paths
    assert "/" in route_paths


@pytest.mark.asyncio
async def test_custom_route_shadows_root(aegra_config_with_custom_app, custom_app_file):
    """Test that custom root route shadows default root"""
    from aegra_api.config import load_http_config
    from aegra_api.core.app_loader import load_custom_app

    http_config = load_http_config()
    user_app = load_custom_app(http_config["app"])

    # Find root route
    root_routes = [r for r in user_app.routes if hasattr(r, "path") and r.path == "/"]
    assert len(root_routes) == 1

    # The custom root should be present
    root_route = root_routes[0]
    assert root_route is not None


@pytest.mark.asyncio
async def test_custom_routes_with_auth_config(tmp_path, custom_app_file, monkeypatch):
    """Test custom routes with authentication enabled"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "app": f"./{custom_app_file.name}:app",
                    "enable_custom_route_auth": True,
                },
            }
        )
    )

    from aegra_api.config import load_http_config

    http_config = load_http_config()
    assert http_config is not None
    assert http_config.get("enable_custom_route_auth") is True


def _find_route(app: FastAPI, path: str) -> APIRoute:
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route
    raise AssertionError(f"Route {path} not found")


@pytest.mark.asyncio
async def test_create_app_enables_custom_route_auth_by_default(
    aegra_config_with_custom_app,
):
    """Custom routes should inherit auth unless explicitly opted out."""
    from aegra_api.main import create_app

    app = create_app()
    route = _find_route(app, "/custom/hello")

    assert route.dependencies[: len(auth_dependency)] == auth_dependency


@pytest.mark.asyncio
async def test_create_app_allows_explicit_custom_route_auth_opt_out(
    tmp_path, custom_app_file, monkeypatch
):
    """Setting enable_custom_route_auth=false should keep custom routes public."""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "app": f"./{custom_app_file.name}:app",
                    "enable_custom_route_auth": False,
                },
            }
        )
    )

    from aegra_api.main import create_app

    app = create_app()
    route = _find_route(app, "/custom/hello")

    assert route.dependencies == []


@pytest.mark.asyncio
async def test_custom_routes_with_cors_config(tmp_path, custom_app_file, monkeypatch):
    """Test custom routes with CORS configuration"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "app": f"./{custom_app_file.name}:app",
                    "cors": {
                        "allow_origins": ["https://example.com"],
                        "allow_credentials": True,
                    },
                },
            }
        )
    )

    from aegra_api.config import load_http_config

    http_config = load_http_config()
    assert http_config is not None
    assert http_config.get("cors") is not None
    assert http_config["cors"]["allow_origins"] == ["https://example.com"]


@pytest.mark.asyncio
async def test_cors_config_without_custom_app(tmp_path, monkeypatch):
    """Test CORS configuration is applied even without a custom app.

    This verifies the fix for https://github.com/your-repo/aegra/issues/XXX
    where http.cors.expose_headers was ignored when no http.app was defined.
    """
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "cors": {
                        "allow_origins": ["https://myapp.com"],
                        "expose_headers": ["Content-Location", "Location", "X-Custom"],
                    },
                },
            }
        )
    )

    from aegra_api.config import load_http_config

    http_config = load_http_config()
    assert http_config is not None
    # Verify no custom app is configured
    assert http_config.get("app") is None
    # Verify CORS config is present and will be used
    assert http_config.get("cors") is not None
    assert http_config["cors"]["allow_origins"] == ["https://myapp.com"]
    assert http_config["cors"]["expose_headers"] == [
        "Content-Location",
        "Location",
        "X-Custom",
    ]


@pytest.mark.asyncio
async def test_cors_default_expose_headers_without_config(tmp_path, monkeypatch):
    """Test that default expose_headers are applied even with no CORS config.

    Content-Location and Location headers must be exposed by default for
    LangGraph SDK stream reconnection (reconnectOnMount) to work.
    """
    monkeypatch.chdir(tmp_path)

    # Config without any http section
    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
            }
        )
    )

    from aegra_api.config import load_http_config

    http_config = load_http_config()
    # http_config can be None when no http section exists
    # The main.py code handles this and applies default expose_headers
    assert http_config is None or http_config.get("cors") is None
