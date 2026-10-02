"""Post a GitHub release to Discord through a channel webhook.

Usage: python scripts/notify_discord_release.py <release.json>
where release.json is `gh release view <tag> --json name,url,body`.
Reads DISCORD_WEBHOOK_URL and PACKAGE (aegra-api, aegra-cli or both) from the env.
"""

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Embed description limit is 4096; leave room for the install line and the overflow note.
_DESCRIPTION_BUDGET = 3800
_EMBED_COLOR = 0x534AB7

# GitHub's generated notes; PR titles are Conventional Commits (enforced by conventional-commits.yml).
_CHANGE_LINE = re.compile(
    r"^\*\s+(?P<type>[a-z]+)(?:\([^)]*\))?(?P<breaking>!)?:\s*(?P<subject>.+?)"
    r"\s+by\s+@(?P<author>\S+)\s+in\s+(?P<url>\S+/pull/(?P<number>\d+))\s*$"
)
_NEW_CONTRIBUTOR_LINE = re.compile(
    r"^\*\s+@(?P<author>\S+)\s+made their first contribution in\s+\S+/pull/(?P<number>\d+)"
)
_IDENTIFIER = re.compile(r"\b\w+_\w+\b")

# Types not listed here (chore, ci, test, refactor, ...) are internal and left out of the post.
_SECTIONS: list[tuple[str, frozenset[str]]] = [
    ("✨ New", frozenset({"feat"})),
    ("🐛 Fixes", frozenset({"fix", "perf"})),
    ("📚 Docs", frozenset({"docs"})),
]

_INSTALL_COMMANDS: dict[str, str] = {
    "aegra-api": "pip install -U aegra-api",
    "aegra-cli": "pip install -U aegra-cli",
    "both": "pip install -U aegra-api aegra-cli",
}


@dataclass(frozen=True)
class Change:
    type: str
    breaking: bool
    subject: str
    author: str
    number: str
    url: str

    def line(self) -> str:
        subject = f"{self.subject[:1].upper()}{self.subject[1:]}"
        if "`" not in subject:
            # snake_case words in PR titles are almost always API names
            subject = _IDENTIFIER.sub(r"`\g<0>`", subject)
        return f"• {subject} ([#{self.number}]({self.url}))"


def parse_changes(body: str) -> list[Change]:
    changes: list[Change] = []
    for line in body.splitlines():
        match = _CHANGE_LINE.match(line.strip())
        if match is None:
            continue
        changes.append(
            Change(
                type=match["type"],
                breaking=match["breaking"] is not None,
                subject=match["subject"],
                author=match["author"],
                number=match["number"],
                url=match["url"],
            )
        )
    return changes


def _sections(changes: list[Change]) -> list[tuple[str, list[Change]]]:
    sections = [("⚠️ Breaking", [c for c in changes if c.breaking])]
    sections += [(title, [c for c in changes if not c.breaking and c.type in types]) for title, types in _SECTIONS]
    return [(title, items) for title, items in sections if items]


def _join_names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def build_payload(*, name: str, url: str, body: str, package: str) -> dict[str, Any]:
    install = f"```\n{_INSTALL_COMMANDS.get(package, _INSTALL_COMMANDS['both'])}\n```"
    sections = _sections(parse_changes(body))
    shown = [change for _, items in sections for change in items]
    newcomers = [m for m in (_NEW_CONTRIBUTOR_LINE.match(line.strip()) for line in body.splitlines()) if m is not None]
    # Newcomers get their own line below, so they are not thanked twice.
    new_authors = {m["author"] for m in newcomers}
    thanked = [author for author in dict.fromkeys(change.author for change in shown) if author not in new_authors]
    credits = [f"🙌 Thanks {_join_names(thanked)}"] if thanked else []
    credits += [f"🎉 First contribution from {m['author']} in #{m['number']}" for m in newcomers]
    if credits:
        credits.append("")

    entries: list[tuple[str, bool]] = []
    for title, items in sections:
        entries += [(f"**{title}**", False), *((change.line(), True) for change in items), ("", False)]

    lines: list[str] = []
    used = len(install) + sum(len(line) + 1 for line in credits)
    listed = 0
    for text, is_change in entries:
        if used + len(text) + 1 > _DESCRIPTION_BUDGET:
            lines += [f"…and {len(shown) - listed} more in the release notes", ""]
            break
        lines.append(text)
        used += len(text) + 1
        listed += is_change

    description = "\n".join([*lines, *credits, install]).strip()
    embed: dict[str, Any] = {
        "title": f"🚀 {name} is out",
        "url": url,
        "description": description,
        "color": _EMBED_COLOR,
    }
    if shown:
        authors = len({change.author for change in shown})
        embed["footer"] = {
            "text": f"{len(shown)} change{'s' * (len(shown) != 1)} from {authors} contributor{'s' * (authors != 1)}"
        }
    return {
        "embeds": [embed],
        # PR titles are user-controlled; never let one ping @everyone or a role.
        "allowed_mentions": {"parse": []},
    }


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """urllib replays a redirected POST as a bodyless GET, which would report success with nothing posted."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def post(webhook_url: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode(),
        # Discord's edge rejects the default urllib user agent with a 403.
        headers={"Content-Type": "application/json", "User-Agent": "aegra-release-notifier"},
        method="POST",
    )
    opener = urllib.request.build_opener(_RejectRedirects())
    with opener.open(request, timeout=30) as response:
        print(f"Discord responded {response.status}")


def main(release_path: str) -> None:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL is not set; skipping the Discord post")
        return

    release = json.loads(Path(release_path).read_text(encoding="utf-8"))
    payload = build_payload(
        name=release["name"],
        url=release["url"],
        body=release.get("body") or "",
        package=os.environ.get("PACKAGE", "both"),
    )
    post(webhook_url, payload)


if __name__ == "__main__":
    main(sys.argv[1])
