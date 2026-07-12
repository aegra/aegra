"""API 响应中的密钥脱敏。

仅用于回传给客户端的副本(如 assistant 的 config/context 可能含自定义
OpenAI 的 api_key)。**不修改存储值**——graph 执行仍读数据库里的原始值。
"""

from typing import Any

_REDACTED = "***"

# 键名规范化(小写、去 - 与 _)后命中即脱敏。覆盖常见密钥/令牌字段。
_SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "password",
        "secret",
        "token",
        "accesstoken",
        "refreshtoken",
        "authorization",
        "cookie",
        "clientsecret",
    }
)


def _is_sensitive(key: str) -> bool:
    """键名规范化后是否命中敏感集合。"""
    return key.lower().replace("-", "").replace("_", "") in _SENSITIVE_KEYS


def redact_secrets(value: Any) -> Any:
    """递归返回脱敏副本:敏感键的值替换为 '***'。不修改入参。"""
    if isinstance(value, dict):
        return {k: (_REDACTED if isinstance(k, str) and _is_sensitive(k) else redact_secrets(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value
