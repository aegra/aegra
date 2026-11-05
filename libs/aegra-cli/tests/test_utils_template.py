"""Tests for chat model loading in generated projects."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock

import pytest

from aegra_cli.templates import render_shared_template_file

type LoadChatModel = Callable[[str], object]


@pytest.fixture
def rendered_load_chat_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LoadChatModel, MagicMock]:
    """Load the rendered utility with its LangChain imports stubbed."""
    init_chat_model = MagicMock(return_value=object())

    langchain = ModuleType("langchain")
    chat_models = ModuleType("langchain.chat_models")
    setattr(chat_models, "init_chat_model", init_chat_model)

    langchain_core = ModuleType("langchain_core")
    language_models = ModuleType("langchain_core.language_models")
    setattr(language_models, "BaseChatModel", object)

    monkeypatch.setitem(sys.modules, "langchain", langchain)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", chat_models)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.language_models", language_models)

    module_path = tmp_path / "generated_utils.py"
    module_path.write_text(
        render_shared_template_file("utils.py.template", {"project_name": "Test Project"}),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("generated_utils", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    load_chat_model = cast(LoadChatModel, getattr(module, "load_chat_model"))
    return load_chat_model, init_chat_model


def test_load_chat_model_keeps_slashes_in_model_id(
    rendered_load_chat_model: tuple[LoadChatModel, MagicMock],
) -> None:
    load_chat_model, init_chat_model = rendered_load_chat_model

    result = load_chat_model("openai/deepseek-ai/DeepSeek-V3")

    init_chat_model.assert_called_once_with(
        "deepseek-ai/DeepSeek-V3",
        model_provider="openai",
    )
    assert result is init_chat_model.return_value


@pytest.mark.parametrize(
    "fully_specified_name",
    ["openai/", "/gpt-4o", "gpt-4o"],
)
def test_load_chat_model_rejects_invalid_names(
    rendered_load_chat_model: tuple[LoadChatModel, MagicMock],
    fully_specified_name: str,
) -> None:
    load_chat_model, init_chat_model = rendered_load_chat_model

    with pytest.raises(ValueError, match="provider/model"):
        load_chat_model(fully_specified_name)

    init_chat_model.assert_not_called()
