"""Utility & helper functions."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

_ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    if provider == "orcarouter":
        # OrcaRouter model IDs may themselves contain a slash (e.g. deepseek/deepseek-v4-pro).
        api_key = os.environ.get("ORCAROUTER_API_KEY")
        if not api_key:
            raise ValueError("ORCAROUTER_API_KEY is not set — add it to your .env to use OrcaRouter.")
        return ChatOpenAI(model=model, api_key=api_key, base_url=_ORCAROUTER_BASE_URL)
    return init_chat_model(model, model_provider=provider)
