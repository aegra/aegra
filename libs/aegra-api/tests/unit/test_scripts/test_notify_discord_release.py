import email.message
import importlib.util
import io
import json
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "notify_discord_release.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("notify_discord_release", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


notify = _load_script()

RELEASE_BODY = """## What's Changed
* feat(api): honor durability and checkpoint_during on runs and crons by @arturbagramyan1 in https://github.com/aegra/aegra/pull/645
* fix(deps): upgrade packages with open security advisories by @ibbybuilds in https://github.com/aegra/aegra/pull/662
* chore(release): v0.10.8 by @ibbybuilds in https://github.com/aegra/aegra/pull/663

## New Contributors
* @someone made their first contribution in https://github.com/aegra/aegra/pull/640

**Full Changelog**: https://github.com/aegra/aegra/compare/v0.10.7...v0.10.8"""


def test_parse_changes_keeps_pr_titles_and_numbers() -> None:
    changes = notify.parse_changes(RELEASE_BODY)

    assert changes == [
        "• feat(api): honor durability and checkpoint_during on runs and crons (#645)",
        "• fix(deps): upgrade packages with open security advisories (#662)",
    ]


def test_parse_changes_skips_release_commit_and_contributor_lines() -> None:
    changes = notify.parse_changes(RELEASE_BODY)

    assert not any("chore(release)" in line for line in changes)
    assert not any("first contribution" in line for line in changes)


def test_parse_changes_returns_empty_for_empty_body() -> None:
    assert notify.parse_changes("") == []


def test_build_payload_blocks_all_mentions() -> None:
    body = "* fix: ping @everyone by @x in https://github.com/aegra/aegra/pull/1"

    payload = notify.build_payload(name="Aegra v1", url="https://example.test/r", body=body, package="both")

    assert payload["allowed_mentions"] == {"parse": []}


@pytest.mark.parametrize(
    ("package", "command"),
    [
        ("both", "pip install -U aegra-api aegra-cli"),
        ("aegra-api", "pip install -U aegra-api"),
        ("aegra-cli", "pip install -U aegra-cli"),
        ("unknown", "pip install -U aegra-api aegra-cli"),
    ],
)
def test_build_payload_install_line_matches_released_package(package: str, command: str) -> None:
    payload = notify.build_payload(name="Aegra v1", url="https://example.test/r", body=RELEASE_BODY, package=package)

    assert f"```\n{command}\n```" in payload["embeds"][0]["description"]


def test_build_payload_embed_links_to_release() -> None:
    payload = notify.build_payload(
        name="Aegra v0.10.8",
        url="https://github.com/aegra/aegra/releases/tag/v0.10.8",
        body=RELEASE_BODY,
        package="both",
    )

    embed = payload["embeds"][0]
    assert embed["title"] == "Aegra v0.10.8"
    assert embed["url"] == "https://github.com/aegra/aegra/releases/tag/v0.10.8"
    assert embed["description"].startswith("• feat(api): honor durability")


def test_build_payload_with_no_changes_still_has_install_line() -> None:
    payload = notify.build_payload(name="Aegra v1", url="https://example.test/r", body="", package="both")

    assert payload["embeds"][0]["description"] == "```\npip install -U aegra-api aegra-cli\n```"


def test_build_payload_reports_overflow_instead_of_cutting_silently() -> None:
    body = "\n".join(
        f"* fix: change number {i} {'x' * 80} by @x in https://github.com/aegra/aegra/pull/{i}" for i in range(100)
    )

    description = notify.build_payload(name="Aegra v1", url="https://example.test/r", body=body, package="both")[
        "embeds"
    ][0]["description"]

    assert len(description) <= 4096
    assert "more in the release notes" in description
    assert description.endswith("```\npip install -U aegra-api aegra-cli\n```")


def test_main_skips_without_webhook(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    posted: list[Any] = []
    monkeypatch.setattr(notify, "post", lambda *args: posted.append(args))

    exit_code = notify.main(["notify_discord_release.py", "release.json"])

    assert exit_code == 0
    assert posted == []
    assert "skipping" in capsys.readouterr().out


def test_main_rejects_missing_release_file_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/api/webhooks/1/x")

    assert notify.main(["notify_discord_release.py"]) == 2


def test_main_posts_payload_built_from_release_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release_file = tmp_path / "release.json"
    release_file.write_text(
        json.dumps({"name": "Aegra v0.10.8", "url": "https://example.test/r", "body": RELEASE_BODY}), encoding="utf-8"
    )
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/api/webhooks/1/x")
    monkeypatch.setenv("PACKAGE", "aegra-api")
    posted: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(notify, "post", lambda url, payload: posted.append((url, payload)))

    exit_code = notify.main(["notify_discord_release.py", str(release_file)])

    assert exit_code == 0
    assert posted[0][0] == "https://discord.test/api/webhooks/1/x"
    assert "pip install -U aegra-api\n" in posted[0][1]["embeds"][0]["description"]


@pytest.mark.parametrize("url", ["http://discord.test/api/webhooks/1/x", "file:///etc/passwd", "discord.test/x"])
def test_post_rejects_non_https_webhook(url: str) -> None:
    with pytest.raises(ValueError, match="https"):
        notify.post(url, {"embeds": []})


class _FakeDiscord(urllib.request.BaseHandler):
    handler_order = 100  # ahead of the default HTTPSHandler, so no request leaves the process

    def __init__(self, status: int, location: str | None = None) -> None:
        self.status = status
        self.location = location
        self.requests: list[urllib.request.Request] = []

    def https_open(self, req: urllib.request.Request) -> urllib.response.addinfourl:
        self.requests.append(req)
        headers = email.message.Message()
        if self.location:
            headers["Location"] = self.location
        response = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, self.status)
        response.msg = "fake"
        return response


def _route_to(monkeypatch: pytest.MonkeyPatch, fake: _FakeDiscord) -> None:
    real_build_opener = urllib.request.build_opener
    monkeypatch.setattr(notify.urllib.request, "build_opener", lambda *handlers: real_build_opener(*handlers, fake))


def test_post_sends_json_payload(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = _FakeDiscord(status=204)
    _route_to(monkeypatch, fake)

    notify.post("https://discord.test/api/webhooks/1/x", {"embeds": [{"title": "Aegra v1"}]})

    assert fake.requests[0].get_method() == "POST"
    assert json.loads(fake.requests[0].data) == {"embeds": [{"title": "Aegra v1"}]}
    assert "Discord responded 204" in capsys.readouterr().out


def test_post_raises_on_redirect_instead_of_reporting_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDiscord(status=302, location="https://elsewhere.test/")
    _route_to(monkeypatch, fake)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        notify.post("https://discord.test/api/webhooks/1/x", {"embeds": []})

    assert exc_info.value.code == 302
    assert len(fake.requests) == 1
