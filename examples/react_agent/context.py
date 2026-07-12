"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from react_agent import prompts

# Fallback mapping from context field name to env var name (OPENAI_ prefix).
_ENV_ALIASES = {
    "model": "OPENAI_MODEL",
    "base_url": "OPENAI_BASE_URL",
    "api_key": "OPENAI_API_KEY",
}


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    base_url: str | None = field(
        default=None,
        metadata={"description": "Base URL for an OpenAI-compatible endpoint; overrides the OPENAI_BASE_URL env var."},
    )

    api_key: str | None = field(
        default=None,
        metadata={
            "description": "API key overriding OPENAI_API_KEY. Note: storing it in an assistant persists it;"
            " for sensitive use, prefer an environment variable."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "The maximum number of search results to return for each search query."},
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_name = _ENV_ALIASES.get(f.name, f.name.upper())
                setattr(self, f.name, os.environ.get(env_name, f.default))
