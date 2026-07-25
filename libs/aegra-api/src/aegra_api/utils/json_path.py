"""JSON-path navigation for thread search ``extract`` projections.

Supports ``key.subkey``, ``key[0]``, and ``key[-1]`` only — no wildcards or filters.
"""

from __future__ import annotations

import re
from typing import Any

_EXTRACT_PREFIXES: tuple[str, ...] = ("values.", "metadata.", "config.")
_MAX_EXTRACT_PATHS = 10

_SEGMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(-?\d+)\])?$")


def validate_extract_paths(extract: dict[str, str]) -> None:
    """Validate extract map size and each path. Raises ValueError on bad input."""
    if len(extract) > _MAX_EXTRACT_PATHS:
        raise ValueError(f"extract supports at most {_MAX_EXTRACT_PATHS} paths")
    for alias, path in extract.items():
        if not isinstance(alias, str) or not alias:
            raise ValueError("extract keys must be non-empty strings")
        if not isinstance(path, str) or not path:
            raise ValueError("extract path must be a non-empty string")
        validate_extract_path(path)


def validate_extract_path(path: str) -> None:
    """Ensure path has an allowed prefix and only supported navigation syntax."""
    if not path.startswith(_EXTRACT_PREFIXES):
        allowed = ", ".join(_EXTRACT_PREFIXES)
        raise ValueError(f"extract path must start with one of: {allowed}")
    _root, _, remainder = path.partition(".")
    for segment in _split_path_segments(remainder):
        if _SEGMENT_RE.match(segment) is None:
            raise ValueError(f"malformed extract path segment: {segment!r} in {path!r}")


def _split_path_segments(remainder: str) -> list[str]:
    """Split ``a.b[0].c`` into ``['a', 'b[0]', 'c']``."""
    if not remainder:
        raise ValueError("extract path must include a field after the prefix")
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(remainder):
        ch = remainder[i]
        if ch == ".":
            if not buf:
                raise ValueError(f"malformed extract path: empty segment in {remainder!r}")
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        if ch == "[":
            buf.append(ch)
            i += 1
            while i < len(remainder) and remainder[i] != "]":
                buf.append(remainder[i])
                i += 1
            if i >= len(remainder) or remainder[i] != "]":
                raise ValueError(f"malformed extract path: unclosed '[' in {remainder!r}")
            buf.append("]")
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def resolve_json_path(root: Any, path: str) -> Any:
    """Resolve a validated path against ``root``. Missing nodes return ``None``."""
    _root_name, _, remainder = path.partition(".")
    current: Any = root
    try:
        segments = _split_path_segments(remainder)
    except ValueError:
        return None
    for segment in segments:
        match = _SEGMENT_RE.match(segment)
        if match is None:
            return None
        key, index_s = match.group(1), match.group(2)
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
        if index_s is not None:
            try:
                idx = int(index_s)
            except ValueError:
                return None
            if not isinstance(current, list | tuple):
                return None
            try:
                current = current[idx]
            except IndexError:
                return None
    return current


def sources_needed_by_extract(extract: dict[str, str]) -> set[str]:
    """Return source names (values/metadata/config) referenced by paths."""
    needed: set[str] = set()
    for path in extract.values():
        root = path.split(".", 1)[0]
        if root in {"values", "metadata", "config"}:
            needed.add(root)
    return needed
