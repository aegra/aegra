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
from pathlib import Path
from typing import Any

# Embed description limit is 4096; leave room for the install line and the overflow note.
_DESCRIPTION_BUDGET = 3800
_EMBED_COLOR = 0x5865F2

# Matches GitHub's generated notes: "* <title> by @<user> in https://github.com/<repo>/pull/<n>"
_CHANGE_LINE = re.compile(r"^\*\s+(?P<title>.+?)\s+by\s+@\S+\s+in\s+\S+/pull/(?P<number>\d+)\s*$")

_INSTALL_COMMANDS: dict[str, str] = {
    "aegra-api": "pip install -U aegra-api",
    "aegra-cli": "pip install -U aegra-cli",
    "both": "pip install -U aegra-api aegra-cli",
}


def parse_changes(body: str) -> list[str]:
    changes: list[str] = []
    for line in body.splitlines():
        match = _CHANGE_LINE.match(line.strip())
        if match is None or match["title"].startswith("chore(release)"):
            continue
        changes.append(f"• {match['title']} (#{match['number']})")
    return changes


def build_payload(*, name: str, url: str, body: str, package: str) -> dict[str, Any]:
    install = f"```\n{_INSTALL_COMMANDS.get(package, _INSTALL_COMMANDS['both'])}\n```"
    changes = parse_changes(body)

    lines: list[str] = []
    used = len(install)
    for index, change in enumerate(changes):
        if used + len(change) + 1 > _DESCRIPTION_BUDGET:
            lines.append(f"…and {len(changes) - index} more in the release notes")
            break
        lines.append(change)
        used += len(change) + 1

    description = "\n".join([*lines, "", install]) if lines else install
    return {
        "embeds": [{"title": name, "url": url, "description": description, "color": _EMBED_COLOR}],
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
    if not webhook_url.startswith("https://"):
        raise ValueError("DISCORD_WEBHOOK_URL must be an https:// URL")
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


def main(argv: list[str]) -> int:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL is not set; skipping the Discord post")
        return 0
    if len(argv) != 2:
        print("usage: notify_discord_release.py <release.json>", file=sys.stderr)
        return 2

    release = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    payload = build_payload(
        name=release["name"],
        url=release["url"],
        body=release.get("body") or "",
        package=os.environ.get("PACKAGE", "both"),
    )
    post(webhook_url, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
