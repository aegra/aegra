"""Unit tests for HTTP and store configuration loading"""

import json
from pathlib import Path

import pytest

from aegra_api.config import (
    load_checkpointer_config,
    load_http_config,
    load_store_config,
    parse_store_ttl_config,
)


def test_load_http_config_from_aegra_json(tmp_path, monkeypatch):
    """Test loading HTTP config from aegra.json"""
    # Change to temp directory
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


def test_load_http_config_from_langgraph_json(tmp_path, monkeypatch):
    """Test loading HTTP config from langgraph.json fallback"""
    # Change to temp directory
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


def test_load_http_config_prefers_aegra_json(tmp_path, monkeypatch):
    """Test that aegra.json takes precedence over langgraph.json"""
    # Change to temp directory
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


def test_load_http_config_no_config(tmp_path, monkeypatch):
    """Test loading when no config file exists"""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    config = load_http_config()

    assert config is None


def test_load_http_config_no_http_section(tmp_path, monkeypatch):
    """Test loading when config exists but no http section"""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(json.dumps({"graphs": {"test": "./test.py:graph"}}))

    config = load_http_config()

    assert config is None


def test_load_http_config_invalid_json(tmp_path, monkeypatch):
    """Test loading when config file has invalid JSON"""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text("{ invalid json }")

    # Should return None and log warning
    config = load_http_config()

    assert config is None


# ============================================================================
# Store Config Tests
# ============================================================================


def test_load_store_config_with_index(tmp_path, monkeypatch):
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


def test_load_store_config_with_ttl(tmp_path, monkeypatch):
    """store.ttl is read from aegra.json alongside the other store keys"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "aegra.json").write_text(json.dumps({"store": {"ttl": {"default_ttl": 90, "refresh_on_read": False}}}))

    config = load_store_config()

    assert config is not None
    assert config["ttl"] == {"default_ttl": 90, "refresh_on_read": False}


def test_parse_store_ttl_config_absent_means_disabled():
    assert parse_store_ttl_config(None) is None


def test_parse_store_ttl_config_applies_defaults():
    """An empty block enables per-item TTLs and the sweeper without a default expiry"""
    assert parse_store_ttl_config({}) == {"refresh_on_read": True}


def test_parse_store_ttl_config_passes_values_through():
    config = parse_store_ttl_config({"refresh_on_read": False, "default_ttl": 1440, "sweep_interval_minutes": 10})

    assert config == {"refresh_on_read": False, "default_ttl": 1440.0, "sweep_interval_minutes": 10.0}


@pytest.mark.parametrize(
    "raw",
    [
        "60",
        [60],
        {"default_ttl": 0},
        {"default_ttl": -1},
        {"default_ttl": "60"},
        {"default_ttl": True},
        {"default_ttl": 10**12},
        {"sweep_interval_minutes": 0},
        {"refresh_on_read": "yes"},
        {"default_ttls": 60},
    ],
)
def test_parse_store_ttl_config_rejects_invalid_blocks(raw):
    """A broken retention policy must fail loudly instead of being ignored"""
    with pytest.raises(ValueError, match="store.ttl"):
        parse_store_ttl_config(raw)


def test_load_store_config_no_config(tmp_path, monkeypatch):
    """Test loading when no config file exists"""
    monkeypatch.chdir(tmp_path)

    config = load_store_config()

    assert config is None


def test_load_store_config_no_store_section(tmp_path, monkeypatch):
    """Test loading when config exists but no store section"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(json.dumps({"graphs": {"test": "./test.py:graph"}}))

    config = load_store_config()

    assert config is None


def test_load_store_config_from_langgraph_json(tmp_path, monkeypatch):
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
# Checkpointer Config Tests
# ============================================================================


def test_load_checkpointer_config_with_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading checkpointer config with a ttl block"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "checkpointer": {"ttl": {"strategy": "delete", "default_ttl": 43200}},
            }
        )
    )

    config = load_checkpointer_config()

    assert config is not None
    assert config["ttl"]["strategy"] == "delete"
    assert config["ttl"]["default_ttl"] == 43200


def test_load_checkpointer_config_no_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when config exists but has no checkpointer section"""
    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(json.dumps({"graphs": {"test": "./test.py:graph"}}))

    config = load_checkpointer_config()

    assert config is None


def test_load_checkpointer_config_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading when no config file exists"""
    monkeypatch.chdir(tmp_path)

    config = load_checkpointer_config()

    assert config is None
