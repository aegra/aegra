"""Envelope encryption for at-rest secrets (e.g. per-assistant API keys).

Fernet (AES-128-CBC + HMAC) keyed by the ``AEGRA_SECRET_KEY`` env var — a
urlsafe base64 32-byte key from ``Fernet.generate_key()``. When the key is
unset, encryption is disabled and encrypt()/decrypt() raise, so a secret is
never written or read in the clear by accident.
"""

from __future__ import annotations

import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

_ENV_KEY = "AEGRA_SECRET_KEY"


@lru_cache(maxsize=1)
def _fernet() -> Fernet | None:
    key = os.environ.get(_ENV_KEY)
    if not key:
        return None
    return Fernet(key.encode())


def is_encryption_enabled() -> bool:
    """Whether AEGRA_SECRET_KEY is configured."""
    return _fernet() is not None


def encrypt(plaintext: str) -> str:
    """Encrypt a secret to a Fernet token. Raises RuntimeError if no key is set."""
    f = _fernet()
    if f is None:
        raise RuntimeError(f"{_ENV_KEY} is not set; refusing to store a secret in the clear")
    return f.encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a Fernet token. Raises RuntimeError if no key, ValueError if tampered."""
    f = _fernet()
    if f is None:
        raise RuntimeError(f"{_ENV_KEY} is not set; cannot decrypt secrets")
    try:
        return f.decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("invalid or tampered secret token") from exc


def encrypt_values(values: dict[str, str]) -> dict[str, str]:
    """Encrypt each value of a mapping to Fernet tokens (raises if no key set)."""
    return {k: encrypt(v) for k, v in values.items()}
