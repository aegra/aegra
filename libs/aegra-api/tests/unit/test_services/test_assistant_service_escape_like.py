"""Regression tests for _escape_like (CWE-89 LIKE-injection fix)."""

import pytest

from aegra_api.services.assistant_service import _escape_like


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain", "plain"),
        ("", ""),
        ("100%", r"100\%"),
        ("a_b", r"a\_b"),
        ("a%b_c", r"a\%b\_c"),
        # Backslash must be escaped first so we don't double-escape the
        # escapes added for % and _.
        ("\\", "\\\\"),
        ("\\%", r"\\\%"),
        ("\\_", r"\\\_"),
        ("100%\\_x", r"100\%\\\_x"),
    ],
)
def test_escape_like_escapes_wildcards_and_backslash(raw: str, expected: str) -> None:
    assert _escape_like(raw) == expected


def test_escape_like_is_idempotent_under_repeated_escape_then_unescape() -> None:
    # Escaping then "unescaping" via SQL escape char "\\" should recover the
    # original string — this is the property the ILIKE escape="\\" relies on.
    raw = "weird%_\\input"
    escaped = _escape_like(raw)
    # Simulate the ILIKE engine consuming '\\' as an escape character.
    unescaped = escaped.replace(r"\%", "%").replace(r"\_", "_").replace("\\\\", "\\")
    assert unescaped == raw
