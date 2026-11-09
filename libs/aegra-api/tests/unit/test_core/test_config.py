"""Unit tests for HTTP and store configuration loading"""

import json
from pathlib import Path

import pytest

from aegra_api.config import (
    get_default_graph_id,
    load_checkpointer_config,
    load_http_config,
    load_store_config,
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


def _write_config(tmp_path: Path, config: dict) -> None:
    (tmp_path / "aegra.json").write_text(json.dumps(config))


class TestDefaultGraphId:
    """get_default_graph_id resolves the fallback for POST /assistants."""

    def test_single_graph_is_the_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """One graph is unambiguous, so it needs no configuration to be the default."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"graphs": {"agent": "./agent.py:graph"}})

        assert get_default_graph_id() == "agent"

    def test_several_graphs_have_no_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a choice to make, the server does not make it — graph_id stays required."""
        monkeypatch.chdir(tmp_path)
        _write_config(
            tmp_path,
            {"graphs": {"agent": "./agent.py:graph", "other": "./other.py:graph"}},
        )

        assert get_default_graph_id() is None

    def test_configured_default_selects_one_of_several_graphs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """default_graph_id nominates the house default for a multi-graph deployment."""
        monkeypatch.chdir(tmp_path)
        _write_config(
            tmp_path,
            {
                "default_graph_id": "other",
                "graphs": {"agent": "./agent.py:graph", "other": "./other.py:graph"},
            },
        )

        assert get_default_graph_id() == "other"

    def test_configured_default_wins_over_the_sole_graph(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit key is honoured even where the single-graph rule would also fire."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"default_graph_id": "agent", "graphs": {"agent": "./agent.py:graph"}})

        assert get_default_graph_id() == "agent"

    def test_should_raise_when_default_names_no_configured_graph(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo fails loudly, naming both the bad value and what was available."""
        monkeypatch.chdir(tmp_path)
        _write_config(
            tmp_path,
            {
                "default_graph_id": "agnet",
                "graphs": {"agent": "./agent.py:graph", "other": "./other.py:graph"},
            },
        )

        with pytest.raises(ValueError) as exc_info:
            get_default_graph_id()

        message = str(exc_info.value)
        assert "agnet" in message
        assert "agent, other" in message

    def test_should_raise_when_default_is_not_a_string(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-string default can never name a graph."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"default_graph_id": ["agent"], "graphs": {"agent": "./agent.py:graph"}})

        with pytest.raises(ValueError):
            get_default_graph_id()

    def test_no_config_file_has_no_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing to resolve from means graph_id stays required."""
        monkeypatch.chdir(tmp_path)

        assert get_default_graph_id() is None

    def test_langgraph_json_fallback_resolves_a_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The compatibility config file is read the same way."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "langgraph.json").write_text(json.dumps({"graphs": {"agent": "./agent.py:graph"}}))

        assert get_default_graph_id() == "agent"
