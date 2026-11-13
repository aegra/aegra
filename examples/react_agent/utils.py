"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


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
        fully_specified_name (str): String in the format 'provider/model'. The
            model ID may contain slashes, e.g. 'openai/deepseek-ai/DeepSeek-V3'.

    Raises:
        ValueError: If the provider or model is missing or whitespace-only.
    """
    provider, _, model = fully_specified_name.partition("/")
    if not provider.strip() or not model.strip():
        raise ValueError(f"fully_specified_name must be in 'provider/model' format, got '{fully_specified_name}'")
    return init_chat_model(model, model_provider=provider)
