"""Resolve ``store.index`` from aegra.json into the index config the Postgres store takes."""

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import structlog
from langchain_core.embeddings import Embeddings
from langgraph.store.base.embed import AEmbeddingsFunc, EmbeddingsFunc
from langgraph.store.postgres.base import PostgresIndexConfig

logger = structlog.get_logger(__name__)

_EMBED_KWARGS = "embed_kwargs"


def is_embed_path(embed: str) -> bool:
    """Tell a ``path/to/file.py:name`` embed from a ``<provider>:<model-id>`` string."""
    file_part, separator, _ = embed.rpartition(":")
    return bool(separator) and file_part.endswith(".py")


def resolve_store_index(index: Mapping[str, Any], *, base_dir: Path | None) -> PostgresIndexConfig:
    """Load a file-path ``embed`` and apply ``embed_kwargs`` to a provider string.

    A relative path resolves against ``base_dir`` (the config file's directory), else the
    current directory. A provider string without ``embed_kwargs`` is left for LangGraph.

    Raises:
        ValueError: If ``embed_kwargs`` is not an object or is not paired with a provider string.
    """
    embed = index.get("embed")
    has_kwargs = _EMBED_KWARGS in index

    if isinstance(embed, str) and is_embed_path(embed):
        if has_kwargs:
            raise ValueError(
                f"store.index.embed_kwargs applies to a provider string, not to the embedding function {embed!r}; "
                "configure the function in its own file instead"
            )
        return {**_without_kwargs(index), "embed": load_embed(embed, base_dir=base_dir)}

    if not has_kwargs:
        return {**index}

    if not isinstance(embed, str):
        raise ValueError(
            "store.index.embed_kwargs needs store.index.embed to be a provider string "
            "such as 'openai:text-embedding-3-small'"
        )
    embed_kwargs = index.get(_EMBED_KWARGS)
    if not isinstance(embed_kwargs, Mapping):
        raise ValueError(f"store.index.embed_kwargs must be a JSON object, got {type(embed_kwargs).__name__}")
    return {**_without_kwargs(index), "embed": _init_embeddings(embed, embed_kwargs)}


def load_embed(embed: str, *, base_dir: Path | None) -> Embeddings | EmbeddingsFunc | AEmbeddingsFunc:
    """Import the ``Embeddings`` instance or embedding function a ``path/to/file.py:name`` names.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the file defines no such name.
        TypeError: If the name is neither an ``Embeddings`` instance nor callable.
    """
    file_part, _, name = embed.rpartition(":")
    path = Path(file_part)
    if not path.is_absolute():
        path = ((base_dir or Path.cwd()) / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"store.index.embed file not found: {path}")

    module_name = f"aegra_store_embed_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load store.index.embed module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before running so classes defined in the file can resolve their own module.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    if not hasattr(module, name):
        raise AttributeError(f"store.index.embed: {name!r} not found in {path}")
    loaded = getattr(module, name)
    if not isinstance(loaded, Embeddings) and not callable(loaded):
        raise TypeError(
            f"store.index.embed {embed!r} must name an Embeddings instance or an embedding function, "
            f"got {type(loaded).__name__}"
        )

    logger.info(f"Loaded store embeddings {name!r} from {path}")
    return loaded


def _without_kwargs(index: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in index.items() if key != _EMBED_KWARGS}


def _init_embeddings(model: str, kwargs: Mapping[str, Any]) -> Embeddings:
    # langchain is optional for aegra-api; LangGraph needs it for any provider string too.
    try:
        from langchain.embeddings import init_embeddings
    except ImportError as exc:
        raise ImportError(
            "store.index.embed_kwargs needs the langchain package and the provider's integration package, "
            "e.g. `pip install langchain langchain-openai`"
        ) from exc
    return init_embeddings(model, **kwargs)
