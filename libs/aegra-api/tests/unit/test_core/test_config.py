"""Unit tests for HTTP, store, and MCP configuration loading"""

import json
from pathlib import Path

import pytest

from aegra_api.config import load_http_config, load_mcp_config, load_store_config


def test_load_http_config_from_aegra_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading HTTP config from aegra.json"""
    monkeypatch.chdir(tmp_path)

    # Create aegra.json with http config
    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "app": "./custom.py:app",
                    "enable_custom_route_auth": True,
                },
            }
        )
    )

    config = load_http_config()

    assert config is not None
    assert config["app"] == "./custom.py:app"
    assert config["enable_custom_route_auth"] is True


def test_load_http_config_from_langgraph_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading HTTP config from langgraph.json fallback"""
    monkeypatch.chdir(tmp_path)

    # Create langgraph.json with http config (no aegra.json)
    config_file = tmp_path / "langgraph.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {
                    "app": "./custom.py:app",
                },
            }
        )
    )

    config = load_http_config()

    assert config is not None
    assert config["app"] == "./custom.py:app"


def test_load_http_config_prefers_aegra_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that aegra.json takes precedence over langgraph.json"""
    monkeypatch.chdir(tmp_path)

    # Create both config files
    aegra_config = tmp_path / "aegra.json"
    aegra_config.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {"app": "./aegra_custom.py:app"},
            }
        )
    )

    langgraph_config = tmp_path / "langgraph.json"
    langgraph_config.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "http": {"app": "./langgraph_custom.py:app"},
            }
        )
    )

    config = load_http_config()

    assert config is not None
    assert config["app"] == "./aegra_custom.py:app"


def test_load_http_config_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when no config file exists"""
    monkeypatch.chdir(tmp_path)

    config = load_http_config()

    assert config is None


def test_load_http_config_no_http_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when config exists but no http section"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(json.dumps({"graphs": {"test": "./test.py:graph"}}))

    config = load_http_config()

    assert config is None


def test_load_http_config_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when config file has invalid JSON"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text("{ invalid json }")

    # Should return None and log warning
    config = load_http_config()

    assert config is None


# ============================================================================
# Store Config Tests
# ============================================================================


def test_load_store_config_with_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading store config with index configuration"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "store": {
                    "index": {
                        "dims": 1536,
                        "embed": "openai:text-embedding-3-small",
                    }
                },
            }
        )
    )

    config = load_store_config()

    assert config is not None
    assert config["index"]["dims"] == 1536
    assert config["index"]["embed"] == "openai:text-embedding-3-small"


def test_load_store_config_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when no config file exists"""
    monkeypatch.chdir(tmp_path)

    config = load_store_config()

    assert config is None


def test_load_store_config_no_store_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when config exists but no store section"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(json.dumps({"graphs": {"test": "./test.py:graph"}}))

    config = load_store_config()

    assert config is None


def test_load_store_config_from_langgraph_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading store config from langgraph.json fallback"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "langgraph.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "store": {
                    "index": {
                        "dims": 768,
                        "embed": "cohere:embed-english-v3.0",
                    }
                },
            }
        )
    )

    config = load_store_config()

    assert config is not None
    assert config["index"]["dims"] == 768
    assert config["index"]["embed"] == "cohere:embed-english-v3.0"


# ============================================================================
# MCP Config Tests
# ============================================================================


def test_load_mcp_config_with_final_response_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading MCP config with final_response_only set."""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "mcp": {"final_response_only": True},
            }
        )
    )

    config = load_mcp_config()

    assert config is not None
    assert config["final_response_only"] is True


def test_load_mcp_config_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when no config file exists."""
    monkeypatch.chdir(tmp_path)

    config = load_mcp_config()

    assert config is None


def test_load_mcp_config_no_mcp_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when config exists but no mcp section."""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(json.dumps({"graphs": {"test": "./test.py:graph"}}))

    config = load_mcp_config()

    assert config is None


def test_load_mcp_config_from_langgraph_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading MCP config from langgraph.json fallback."""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "langgraph.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "mcp": {"final_response_only": True},
            }
        )
    )

    config = load_mcp_config()

    assert config is not None
    assert config["final_response_only"] is True
