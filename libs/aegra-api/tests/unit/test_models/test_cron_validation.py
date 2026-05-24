"""Validation tests for cron Pydantic models.

Covers reviewer-requested guards: webhook scheme, payload size, end_time
must be in the future, max_length on string fields, on_run_completed
literal restriction.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aegra_api.models.crons import CronCreate, CronUpdate


class TestWebhookValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/hook",
            "javascript:alert(1)",
            "file:///etc/passwd",
            "//example.com/hook",
            "://no-scheme.example",
            "http:///no-host",
        ],
    )
    def test_rejects_non_http_or_malformed_scheme(self, url: str) -> None:
        with pytest.raises(ValidationError):
            CronCreate(assistant_id="a", schedule="* * * * *", webhook=url)

    @pytest.mark.parametrize("url", ["http://example.com/hook", "https://example.com/hook"])
    def test_accepts_http_https(self, url: str) -> None:
        req = CronCreate(assistant_id="a", schedule="* * * * *", webhook=url)
        assert req.webhook == url


class TestEndTimeMustBeFuture:
    def test_rejects_past_end_time_on_create(self) -> None:
        with pytest.raises(ValidationError, match="future"):
            CronCreate(
                assistant_id="a",
                schedule="* * * * *",
                end_time=datetime.now(UTC) - timedelta(seconds=1),
            )

    def test_accepts_future_end_time_on_create(self) -> None:
        end = datetime.now(UTC) + timedelta(days=1)
        req = CronCreate(assistant_id="a", schedule="* * * * *", end_time=end)
        assert req.end_time == end

    def test_rejects_past_end_time_on_update(self) -> None:
        with pytest.raises(ValidationError, match="future"):
            CronUpdate(end_time=datetime.now(UTC) - timedelta(days=365))


class TestPayloadSizeCap:
    def test_rejects_oversized_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aegra_api.models import crons as crons_mod

        monkeypatch.setattr(crons_mod.settings.cron, "CRON_MAX_PAYLOAD_BYTES", 256)
        with pytest.raises(ValidationError, match="payload"):
            CronCreate(
                assistant_id="a",
                schedule="* * * * *",
                input={"messages": [{"role": "user", "content": "x" * 1024}]},
            )


class TestOnRunCompletedLiteral:
    def test_rejects_unknown_value(self) -> None:
        with pytest.raises(ValidationError):
            CronCreate(
                assistant_id="a",
                schedule="* * * * *",
                on_run_completed="create_new",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("value", ["delete", "keep"])
    def test_accepts_allowed_values(self, value: str) -> None:
        req = CronCreate(
            assistant_id="a",
            schedule="* * * * *",
            on_run_completed=value,  # type: ignore[arg-type]
        )
        assert req.on_run_completed == value


class TestMaxLengthGuards:
    def test_rejects_oversized_schedule(self) -> None:
        with pytest.raises(ValidationError):
            CronCreate(assistant_id="a", schedule="*" * 1024)

    def test_rejects_oversized_timezone(self) -> None:
        with pytest.raises(ValidationError):
            CronCreate(assistant_id="a", schedule="* * * * *", timezone="X" * 256)

    def test_rejects_oversized_webhook(self) -> None:
        with pytest.raises(ValidationError):
            CronCreate(
                assistant_id="a",
                schedule="* * * * *",
                webhook="https://example.com/" + ("x" * 4096),
            )
