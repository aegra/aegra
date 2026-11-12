"""Unit tests for resolving store.index embeddings from aegra.json"""

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langgraph.store.base.embed import ensure_embeddings

from aegra_api.config import StoreIndexConfig
from aegra_api.core.store_index import is_embed_path, resolve_store_index

ASYNC_EMBED_FILE = """
async def embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), 1.0] for text in texts]
"""

SYNC_EMBED_FILE = """
def embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), 2.0] for text in texts]
"""

EMBEDDINGS_INSTANCE_FILE = """
from langchain_core.embeddings import Embeddings


class LengthEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 3.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 3.0]


embeddings = LengthEmbeddings()
"""


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


@pytest.mark.parametrize(
    ("embed", "expected"),
    [
        ("./embeddings.py:embed", True),
        ("path/to/embedding_function.py:embed", True),
        ("embeddings.py:embed", True),
        ("C:\\project\\embeddings.py:embed", True),
        ("openai:text-embedding-3-small", False),
        ("bedrock:amazon.titan-embed-text-v2:0", False),
        ("ollama:qwen3-embedding:0.6b", False),
        ("./embeddings.py", False),
    ],
)
def test_is_embed_path_tells_paths_from_provider_strings(embed: str, expected: bool) -> None:
    assert is_embed_path(embed) is expected


def test_provider_string_passes_through_unchanged() -> None:
    index: StoreIndexConfig = {"dims": 1536, "embed": "openai:text-embedding-3-small", "fields": ["$"]}

    resolved = resolve_store_index(index, base_dir=None)

    assert resolved == index


def test_provider_string_keeps_unlisted_index_keys() -> None:
    index: dict[str, Any] = {"dims": 1536, "embed": "openai:text-embedding-3-small", "distance_type": "cosine"}

    resolved = resolve_store_index(index, base_dir=None)

    assert resolved == index


async def test_loads_async_function_relative_to_config_dir(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "embeddings.py", ASYNC_EMBED_FILE)
    index: StoreIndexConfig = {"dims": 2, "embed": "./src/embeddings.py:embed", "fields": ["text"]}

    resolved = resolve_store_index(index, base_dir=tmp_path)

    embeddings = ensure_embeddings(resolved["embed"])
    assert await embeddings.aembed_documents(["abc"]) == [[3.0, 1.0]]
    assert resolved["dims"] == 2
    assert resolved["fields"] == ["text"]


async def test_resolves_relative_path_against_config_dir_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "config" / "path" / "to" / "embedding_function.py", ASYNC_EMBED_FILE)
    _write(tmp_path / "path" / "to" / "embedding_function.py", SYNC_EMBED_FILE)
    monkeypatch.chdir(tmp_path)
    index: StoreIndexConfig = {"dims": 2, "embed": "path/to/embedding_function.py:embed"}

    resolved = resolve_store_index(index, base_dir=tmp_path / "config")

    embeddings = ensure_embeddings(resolved["embed"])
    assert await embeddings.aembed_documents(["abcd"]) == [[4.0, 1.0]]


def test_relative_path_falls_back_to_cwd_without_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / "embeddings.py", SYNC_EMBED_FILE)
    monkeypatch.chdir(tmp_path)
    index: StoreIndexConfig = {"dims": 2, "embed": "./embeddings.py:embed"}

    resolved = resolve_store_index(index, base_dir=None)

    embeddings = ensure_embeddings(resolved["embed"])
    assert embeddings.embed_documents(["ab"]) == [[2.0, 2.0]]


def test_loads_absolute_path(tmp_path: Path) -> None:
    embed_file = _write(tmp_path / "embeddings.py", SYNC_EMBED_FILE)
    index: StoreIndexConfig = {"dims": 2, "embed": f"{embed_file.resolve()}:embed"}

    resolved = resolve_store_index(index, base_dir=tmp_path / "elsewhere")

    embeddings = ensure_embeddings(resolved["embed"])
    assert embeddings.embed_documents(["a"]) == [[1.0, 2.0]]


def test_loads_embeddings_instance(tmp_path: Path) -> None:
    _write(tmp_path / "embeddings.py", EMBEDDINGS_INSTANCE_FILE)
    index: StoreIndexConfig = {"dims": 2, "embed": "./embeddings.py:embeddings"}

    resolved = resolve_store_index(index, base_dir=tmp_path)

    embed = resolved["embed"]
    assert isinstance(embed, Embeddings)
    assert embed.embed_query("abc") == [3.0, 3.0]


def test_path_keeps_other_index_keys(tmp_path: Path) -> None:
    _write(tmp_path / "embeddings.py", SYNC_EMBED_FILE)
    index: dict[str, Any] = {
        "dims": 2,
        "embed": "./embeddings.py:embed",
        "fields": ["text"],
        "distance_type": "inner_product",
    }

    resolved = resolve_store_index(index, base_dir=tmp_path)

    assert resolved.get("distance_type") == "inner_product"
    assert resolved["fields"] == ["text"]
    assert resolved["dims"] == 2


def test_missing_file_raises_with_resolved_path(tmp_path: Path) -> None:
    index: StoreIndexConfig = {"dims": 2, "embed": "./missing.py:embed"}

    with pytest.raises(FileNotFoundError, match="missing.py"):
        resolve_store_index(index, base_dir=tmp_path)


def test_missing_attribute_raises(tmp_path: Path) -> None:
    _write(tmp_path / "embeddings.py", SYNC_EMBED_FILE)
    index: StoreIndexConfig = {"dims": 2, "embed": "./embeddings.py:not_there"}

    with pytest.raises(AttributeError, match="not_there"):
        resolve_store_index(index, base_dir=tmp_path)


def test_non_callable_attribute_raises(tmp_path: Path) -> None:
    _write(tmp_path / "embeddings.py", 'embed = "text-embedding-3-small"\n')
    index: StoreIndexConfig = {"dims": 2, "embed": "./embeddings.py:embed"}

    with pytest.raises(TypeError, match="Embeddings instance or an embedding function"):
        resolve_store_index(index, base_dir=tmp_path)


def test_failed_import_leaves_no_module_behind(tmp_path: Path) -> None:
    _write(tmp_path / "broken_embed.py", "raise RuntimeError('boom')\n")
    index: StoreIndexConfig = {"dims": 2, "embed": "./broken_embed.py:embed"}

    with pytest.raises(RuntimeError, match="boom"):
        resolve_store_index(index, base_dir=tmp_path)

    assert not [name for name in sys.modules if "broken_embed" in name]


def test_embed_kwargs_build_provider_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    index: StoreIndexConfig = {
        "dims": 256,
        "embed": "openai:text-embedding-3-small",
        "embed_kwargs": {"dimensions": 256, "check_embedding_ctx_length": False},
        "fields": ["text"],
    }

    resolved = resolve_store_index(index, base_dir=None)

    embed = resolved["embed"]
    assert isinstance(embed, OpenAIEmbeddings)
    assert embed.model == "text-embedding-3-small"
    assert embed.dimensions == 256
    assert embed.check_embedding_ctx_length is False
    assert "embed_kwargs" not in resolved
    assert resolved["dims"] == 256
    assert resolved["fields"] == ["text"]


def test_empty_embed_kwargs_build_default_provider_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    index: StoreIndexConfig = {"dims": 1536, "embed": "openai:text-embedding-3-small", "embed_kwargs": {}}

    resolved = resolve_store_index(index, base_dir=None)

    embed = resolved["embed"]
    assert isinstance(embed, OpenAIEmbeddings)
    assert embed.check_embedding_ctx_length is True
    assert "embed_kwargs" not in resolved


def test_embed_kwargs_with_path_raises(tmp_path: Path) -> None:
    _write(tmp_path / "embeddings.py", SYNC_EMBED_FILE)
    index: StoreIndexConfig = {"dims": 2, "embed": "./embeddings.py:embed", "embed_kwargs": {"dimensions": 2}}

    with pytest.raises(ValueError, match="embed_kwargs"):
        resolve_store_index(index, base_dir=tmp_path)


@pytest.mark.parametrize("embed_kwargs", [["dimensions", 256], "dimensions=256", 256, None])
def test_embed_kwargs_must_be_an_object(embed_kwargs: object) -> None:
    index: dict[str, Any] = {"dims": 256, "embed": "openai:text-embedding-3-small", "embed_kwargs": embed_kwargs}

    with pytest.raises(ValueError, match="embed_kwargs must be a JSON object"):
        resolve_store_index(index, base_dir=None)


def test_embed_kwargs_without_provider_string_raises() -> None:
    index: StoreIndexConfig = {"dims": 256, "embed_kwargs": {"dimensions": 256}}

    with pytest.raises(ValueError, match="provider string"):
        resolve_store_index(index, base_dir=None)
