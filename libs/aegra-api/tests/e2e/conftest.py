"""E2E test specific fixtures

E2E tests use the full system with real database and services.

Note: Auth tests live in tests/e2e/manual_auth_tests/ and are skipped by default
via pytest.ini (-m "not auth_only"). Run with: make e2e-auth  or  pytest -m auth_only
"""
