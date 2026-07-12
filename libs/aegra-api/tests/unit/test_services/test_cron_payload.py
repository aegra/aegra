"""Tests for cron webhook payload (自定义 headers/body/query) 与 search 过滤字段。"""

from aegra_api.models.crons import CronCountRequest, CronCreate, CronSearchRequest
from aegra_api.services.cron_service import _build_payload, _redact_payload


class TestWebhookPayload:
    """需求 C:cron webhook 支持自定义请求头 / body / query,并存入 payload。"""

    def test_custom_webhook_fields_stored(self) -> None:
        req = CronCreate(
            assistant_id="a1",
            schedule="*/5 * * * *",
            webhook="https://example.com/hook",
            webhook_headers={"X-Token": "abc"},
            webhook_body={"event": "done"},
            webhook_query={"src": "cron"},
        )
        payload = _build_payload(req)
        assert payload["webhook"] == "https://example.com/hook"
        assert payload["webhook_headers"] == {"X-Token": "abc"}
        assert payload["webhook_body"] == {"event": "done"}
        assert payload["webhook_query"] == {"src": "cron"}

    def test_absent_webhook_fields_omitted(self) -> None:
        payload = _build_payload(CronCreate(assistant_id="a1", schedule="*/5 * * * *"))
        assert "webhook_headers" not in payload
        assert "webhook_body" not in payload
        assert "webhook_query" not in payload

    def test_redact_masks_sensitive_headers(self) -> None:
        masked = _redact_payload({"webhook_headers": {"Authorization": "Bearer x", "X-Foo": "bar"}})
        assert masked["webhook_headers"]["Authorization"] == "***"
        assert masked["webhook_headers"]["X-Foo"] == "bar"


class TestSearchTenantUserFilters:
    """需求 B:search/count 请求接受 user_id / tenant_id 过滤字段。"""

    def test_search_accepts_user_and_tenant(self) -> None:
        req = CronSearchRequest(user_id="u1", tenant_id="t1")
        assert req.user_id == "u1"
        assert req.tenant_id == "t1"

    def test_count_accepts_user_and_tenant(self) -> None:
        req = CronCountRequest(user_id="u1", tenant_id="t1")
        assert req.user_id == "u1"
        assert req.tenant_id == "t1"

    def test_filters_default_none(self) -> None:
        req = CronSearchRequest()
        assert req.user_id is None
        assert req.tenant_id is None
