"""Secret redaction for API responses (cron webhook headers).

Masks response copies only; stored values are untouched.
"""

from typing import Any

_REDACTED = "***"

# Redact when the normalized key (lowercased, - and _ stripped) matches. Covers common secret/token fields.
_SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "xapikey",
        "password",
        "secret",
        "token",
        "accesstoken",
        "refreshtoken",
        "authorization",
        "proxyauthorization",
        "cookie",
        "clientsecret",
    }
)


def _is_sensitive(key: str) -> bool:
    """Whether the normalized key hits the sensitive set."""
    return key.lower().replace("-", "").replace("_", "") in _SENSITIVE_KEYS


def redact_secrets(value: Any) -> Any:
    """Return a redacted deep copy: sensitive values become '***'. Input is not mutated."""
    if isinstance(value, dict):
        return {k: (_REDACTED if isinstance(k, str) and _is_sensitive(k) else redact_secrets(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value
