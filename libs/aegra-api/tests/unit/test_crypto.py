"""Tests for at-rest secret encryption (core/crypto.py)."""

from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet

from aegra_api.core import crypto


@pytest.fixture
def enc_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AEGRA_SECRET_KEY", key)
    crypto._fernet.cache_clear()
    yield key
    crypto._fernet.cache_clear()


class TestCrypto:
    def test_roundtrip(self, enc_key: str) -> None:
        token = crypto.encrypt("sk-secret")
        assert token != "sk-secret"
        assert crypto.decrypt(token) == "sk-secret"

    def test_disabled_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AEGRA_SECRET_KEY", raising=False)
        crypto._fernet.cache_clear()
        with pytest.raises(RuntimeError):
            crypto.encrypt("x")
        crypto._fernet.cache_clear()

    def test_decrypt_invalid_token_raises(self, enc_key: str) -> None:
        with pytest.raises(ValueError):
            crypto.decrypt("not-a-valid-token")
