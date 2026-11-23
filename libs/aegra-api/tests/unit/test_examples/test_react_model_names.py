"""Validate the example's model names against real LangChain initialization."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from langchain_openai import ChatOpenAI

EXAMPLES_DIR = Path(__file__).resolve().parents[5] / "examples"


@pytest.fixture
def react_utils(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Use real model constructors with dummy credentials and no inference requests."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    yield importlib.import_module("react_agent.utils")
    for name in [name for name in sys.modules if name.split(".")[0] == "react_agent"]:
        del sys.modules[name]


@pytest.mark.parametrize("model_id", ["gpt-4o-mini", "deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.1-8B-Instruct"])
def test_gateway_model_id_reaches_real_openai_client(react_utils: ModuleType, model_id: str) -> None:
    """Nested gateway routes must survive parsing into the actual OpenAI client."""
    model = react_utils.load_chat_model(f"openai/{model_id}")

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == model_id
    assert str(model.root_client.base_url) == "http://127.0.0.1:9/v1/"


@pytest.mark.parametrize("name", ["openai/", "/gpt-4o", "gpt-4o", "", "   ", "openai/  ", "   /gpt-4o"])
def test_missing_provider_or_model_raises_value_error(react_utils: ModuleType, name: str) -> None:
    """Reject incomplete names before LangChain can infer a provider or default model."""
    with pytest.raises(ValueError, match="provider/model"):
        react_utils.load_chat_model(name)
