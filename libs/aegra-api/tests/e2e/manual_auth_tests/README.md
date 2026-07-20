# Auth E2E Tests

These tests verify authentication and authorization against a real server with JWT mock auth enabled. They are skipped by default (`-m "not auth_only"`) and run automatically in CI via the **E2E (Auth Enabled)** job.

## Purpose

Aegra is designed so users bring their own auth. These tests exercise the auth middleware, `@auth.authenticate` / `@auth.on.*` handlers, custom routes, and per-user thread isolation using [`examples/jwt_mock_auth_example.py`](../../../../../examples/jwt_mock_auth_example.py).

## When to run locally

Run these when changing:

- `libs/aegra-api/src/aegra_api/core/auth_middleware.py`
- `libs/aegra-api/src/aegra_api/core/auth_handlers.py`
- `libs/aegra-api/src/aegra_api/core/auth_deps.py`
- Authorization handler resolution
- Auth-related features or bug fixes

## How to run

### Recommended: Makefile

```bash
make e2e-auth
```

This starts Docker (dev executor + [`aegra.auth.json`](../../../../../aegra.auth.json)), waits for health, runs the auth suite, then tears down.

### Manual

1. Start the server with auth config (from repo root):

   ```bash
   # Full Docker (dev executor + auth config):
   docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.auth.yml up -d

   # Or API process only (Postgres already running):
   AEGRA_CONFIG=aegra.auth.json AUTH_TYPE=custom REDIS_BROKER_ENABLED=false \
     uv run --package aegra-api uvicorn aegra_api.main:app --host 127.0.0.1 --port 2026
   ```

2. Run the tests:

   ```bash
   uv run --package aegra-api pytest libs/aegra-api/tests/e2e/manual_auth_tests/ -v -m auth_only

   # Specific file
   uv run --package aegra-api pytest libs/aegra-api/tests/e2e/manual_auth_tests/test_auth_e2e.py -v -m auth_only
   ```

## Config

Root [`aegra.auth.json`](../../../../../aegra.auth.json) registers:

- Graph: `examples/react_agent`
- Auth: `examples/jwt_mock_auth_example.py:auth`
- Custom HTTP app: `examples/custom_routes_example.py:app`

## Test files

- `test_auth_e2e.py` — Core auth flow (JWT, custom routes, error handling)
- `test_authorization_handlers_e2e.py` — `@auth.on.*` authorization handlers
- `test_thread_user_isolation_e2e.py` — Per-user thread isolation

## Notes

- Tests skip automatically if the server is up but auth is not enabled (`check_server_has_auth`)
- Token format for the mock handler: `mock-jwt-<user_id>-<role>-<team_id>`
- Marker: `@pytest.mark.auth_only` (same pattern as `prod_only`)
