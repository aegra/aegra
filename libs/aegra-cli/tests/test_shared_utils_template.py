"""The shared utils template accepts gateway model ids that contain a slash."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from aegra_cli.cli import cli
from aegra_cli.templates import slugify


@pytest.fixture
def rendered_project(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    # The rendered utils.py imports langchain, which is only in aegra-api's dev group
    pytest.importorskip("langchain")
    project_dir = tmp_path / "slash-model"
    result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "2"])
    assert result.exit_code == 0
    slug = slugify("slash-model")
    monkeypatch.syspath_prepend(str(project_dir / "src"))
    yield slug
    for name in [name for name in sys.modules if name.split(".")[0] == slug]:
        del sys.modules[name]


@pytest.fixture
def captured_model(monkeypatch: pytest.MonkeyPatch, rendered_project: str) -> dict[str, Any]:
    utils = importlib.import_module(f"{rendered_project}.utils")
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, model_provider: str | None = None) -> object:
        captured["model"] = model
        captured["provider"] = model_provider
        return object()

    monkeypatch.setattr(utils, "init_chat_model", fake_init_chat_model)
    return captured


@pytest.mark.parametrize(
    ("fully_specified_name", "provider", "model"),
    [
        ("openai/gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("openai/deepseek-ai/DeepSeek-V3", "openai", "deepseek-ai/DeepSeek-V3"),
        ("openai/meta-llama/Llama-3.1-8B-Instruct", "openai", "meta-llama/Llama-3.1-8B-Instruct"),
    ],
)
def test_load_chat_model_keeps_slashes_in_the_model_id(
    captured_model: dict[str, Any],
    rendered_project: str,
    fully_specified_name: str,
    provider: str,
    model: str,
) -> None:
    utils = importlib.import_module(f"{rendered_project}.utils")

    utils.load_chat_model(fully_specified_name)

    assert captured_model == {"model": model, "provider": provider}


@pytest.mark.parametrize(
    "fully_specified_name", ["openai/", "/gpt-4o", "gpt-4o", "", "   ", "openai/  ", "   /gpt-4o"]
)
def test_load_chat_model_rejects_a_missing_provider_or_model(
    captured_model: dict[str, Any], rendered_project: str, fully_specified_name: str
) -> None:
    utils = importlib.import_module(f"{rendered_project}.utils")

    with pytest.raises(ValueError):
        utils.load_chat_model(fully_specified_name)

    assert captured_model == {}
