"""E2E: a completed run POSTs to the configured webhook with a valid signature.

A local HTTP receiver runs on the host; the server (in Docker) reaches it via
host.docker.internal. The run may error (no live LLM needed) — a webhook fires
on any terminal state, which is exactly what we assert.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from aegra_api.models.webhooks import sign
from aegra_api.settings import settings
from tests.e2e._utils import elog

# whsec_ + base64("e2e-webhook-secret")
_SECRET = "whsec_ZTJlLXdlYmhvb2stc2VjcmV0"


class _Collector(BaseHTTPRequestHandler):
    deliveries: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        _Collector.deliveries.append({"headers": dict(self.headers), "body": body})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        """Silence the default stderr access log."""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_completion_delivers_signed_webhook() -> None:
    _Collector.deliveries = []
    server = HTTPServer(("0.0.0.0", 0), _Collector)  # nosec B104 — test receiver, ephemeral port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # The server (Docker) reaches the host receiver via host.docker.internal.
    webhook_url = f"http://host.docker.internal:{port}/hook"
    try:
        async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=60.0) as client:
            resp = await client.post(
                "/runs",
                json={
                    "assistant_id": "agent",
                    "input": {"messages": [{"role": "user", "content": "ping"}]},
                    "webhook": {
                        "url": webhook_url,
                        "headers": {"X-Source": "aegra-e2e"},
                        "params": {"tag": "e2e"},
                        "secret": _SECRET,
                    },
                },
            )
            assert resp.status_code in (200, 201), resp.text
            elog("created run", resp.json())

        # Poll for the delivery (run executes + completes in the background).
        for _ in range(60):
            if _Collector.deliveries:
                break
            await asyncio.sleep(1)

        assert _Collector.deliveries, "webhook was never delivered"
        delivery = _Collector.deliveries[0]
        headers = {k.lower(): v for k, v in delivery["headers"].items()}
        payload = json.loads(delivery["body"])
        elog("received webhook", {"headers": headers, "payload": payload})

        # Canonical run payload + custom header forwarded.
        assert payload["run_id"]
        assert payload["status"] in ("success", "error", "interrupted")
        assert headers.get("x-source") == "aegra-e2e"

        # Signature verifies against the raw body (Standard Webhooks).
        expected = sign(_SECRET, headers["webhook-id"], int(headers["webhook-timestamp"]), delivery["body"])
        assert headers["webhook-signature"] == expected
    finally:
        server.shutdown()
        server.server_close()
