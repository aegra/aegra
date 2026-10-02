import email.message
import importlib.util
import io
import json
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path
from types import ModuleType

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


def test_parse_changes_keeps_pr_lines_and_drops_release_and_contributor_lines() -> None:
    assert notify.parse_changes(RELEASE_BODY) == [
        "• feat(api): honor durability and checkpoint_during on runs and crons (#645)",
        "• fix(deps): upgrade packages with open security advisories (#662)",
    ]


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
    ],
)
def test_build_payload_install_line_matches_published_package(package: str, command: str) -> None:
    payload = notify.build_payload(name="Aegra v1", url="https://example.test/r", body=RELEASE_BODY, package=package)

    assert payload["embeds"][0]["description"].endswith(f"```\n{command}\n```")


def test_build_payload_reports_overflow_instead_of_cutting_silently() -> None:
    body = "\n".join(
        f"* fix: change {i} {'x' * 80} by @x in https://github.com/aegra/aegra/pull/{i}" for i in range(100)
    )

    payload = notify.build_payload(name="Aegra v1", url="https://example.test/r", body=body, package="both")

    description = payload["embeds"][0]["description"]
    assert len(description) <= 4096
    assert "more in the release notes" in description


def test_main_skips_without_webhook(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    notify.main("release.json")

    assert "skipping" in capsys.readouterr().out


class _FakeDiscord(urllib.request.BaseHandler):
    handler_order = 100  # ahead of the default HTTPSHandler, so no request leaves the process

    def __init__(self, status: int) -> None:
        self.status = status
        self.requests: list[urllib.request.Request] = []

    def https_open(self, req: urllib.request.Request) -> urllib.response.addinfourl:
        self.requests.append(req)
        headers = email.message.Message()
        headers["Location"] = "https://elsewhere.test/"
        response = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, self.status)
        response.msg = "fake"
        return response


def _route(monkeypatch: pytest.MonkeyPatch, fake: _FakeDiscord) -> None:
    real_build_opener = urllib.request.build_opener
    monkeypatch.setattr(notify.urllib.request, "build_opener", lambda *handlers: real_build_opener(*handlers, fake))


def test_post_sends_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDiscord(status=204)
    _route(monkeypatch, fake)

    notify.post("https://discord.test/api/webhooks/1/x", {"embeds": [{"title": "Aegra v1"}]})

    assert fake.requests[0].get_method() == "POST"
    assert json.loads(fake.requests[0].data) == {"embeds": [{"title": "Aegra v1"}]}


def test_post_raises_on_redirect_instead_of_reporting_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDiscord(status=302)
    _route(monkeypatch, fake)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        notify.post("https://discord.test/api/webhooks/1/x", {"embeds": []})

    assert exc_info.value.code == 302
    assert len(fake.requests) == 1
