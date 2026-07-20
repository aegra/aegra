.PHONY: help install dev-install setup-hooks format lint type-check security test test-api test-cli test-cov clean run ci-check openapi e2e-dev e2e-prod e2e-auth e2e-both

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make dev-install   - Install all dependencies + git hooks"
	@echo "  make setup-hooks   - Reinstall git hooks (if needed)"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Run ty type checking"
	@echo "  make security      - Run security checks with bandit"
	@echo "  make test          - Run all tests"
	@echo "  make test-api      - Run aegra-api tests only"
	@echo "  make test-cli      - Run aegra-cli tests only"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make openapi       - Regenerate docs/openapi.json from code"
	@echo "  make ci-check      - Run all CI checks locally"
	@echo "  make e2e-dev       - Run E2E tests in dev mode (no Redis)"
	@echo "  make e2e-prod      - Run E2E tests in prod mode (Redis workers)"
	@echo "  make e2e-auth      - Run auth E2E tests (JWT mock auth enabled)"
	@echo "  make e2e-both      - Run E2E tests in both modes"
	@echo "  make clean         - Clean cache files"
	@echo "  make run           - Run the server"

install:
	uv sync --all-packages --no-dev

dev-install:
	uv sync --all-packages
	@uv run pre-commit install
	@uv run pre-commit install --hook-type commit-msg
	@echo ""
	@echo "Done! Dependencies installed and git hooks set up."

setup-hooks:
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	@echo ""
	@echo "Git hooks reinstalled!"

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

type-check:
	uv run ty check libs/aegra-api/src/ libs/aegra-cli/src/

security:
	uv run bandit -c pyproject.toml -r libs/aegra-api/src/ libs/aegra-cli/src/

test: test-api test-cli

test-api:
	uv run --package aegra-api pytest libs/aegra-api/tests/

test-cli:
	uv run --package aegra-cli pytest libs/aegra-cli/tests/

test-cov:
	uv run --package aegra-api pytest libs/aegra-api/tests/ --cov=libs/aegra-api/src --cov-report=html --cov-report=term
	uv run --package aegra-cli pytest libs/aegra-cli/tests/ --cov=libs/aegra-cli/src --cov-report=term

openapi:
	uv run --package aegra-api python scripts/export_openapi.py

ci-check: format lint
	-uv run ty check libs/aegra-api/src/ libs/aegra-cli/src/
	-uv run bandit -c pyproject.toml -r libs/aegra-api/src/ libs/aegra-cli/src/
	$(MAKE) test
	@echo ""
	@echo "All CI checks completed! (ty and bandit are non-blocking)"

E2E_IGNORE := --ignore=libs/aegra-api/tests/e2e/manual_auth_tests --ignore=libs/aegra-api/tests/e2e/multi_instance

e2e-dev:
	@echo "Starting dev mode (LocalExecutor, no Redis)..."
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "Waiting for server..."; \
	ready=0; \
	for i in $$(seq 1 30); do \
		if curl -s http://localhost:2026/health > /dev/null 2>&1; then ready=1; break; fi; \
		sleep 2; \
	done; \
	if [ "$$ready" = "0" ]; then echo "Server failed to start within 60s"; docker compose -f docker-compose.yml -f docker-compose.dev.yml down; exit 1; fi
	@rc=0; \
	uv run --package aegra-api pytest libs/aegra-api/tests/e2e/ -m "not prod_only" $(E2E_IGNORE) -v --tb=short || rc=$$?; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down; \
	exit $$rc

e2e-prod:
	@echo "Starting prod mode (WorkerExecutor + Redis)..."
	@docker compose up -d
	@echo "Waiting for server..."; \
	ready=0; \
	for i in $$(seq 1 30); do \
		if curl -s http://localhost:2026/health > /dev/null 2>&1; then ready=1; break; fi; \
		sleep 2; \
	done; \
	if [ "$$ready" = "0" ]; then echo "Server failed to start within 60s"; docker compose down; exit 1; fi
	@rc=0; \
	uv run --package aegra-api pytest libs/aegra-api/tests/e2e/ $(E2E_IGNORE) -v --tb=short || rc=$$?; \
	docker compose down; \
	exit $$rc

e2e-auth:
	@echo "Starting auth mode (JWT mock auth, LocalExecutor)..."
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.auth.yml up -d postgres aegra
	@echo "Waiting for server..."; \
	ready=0; \
	for i in $$(seq 1 45); do \
		if curl -s http://localhost:2026/health > /dev/null 2>&1; then ready=1; break; fi; \
		sleep 2; \
	done; \
	if [ "$$ready" = "0" ]; then \
		echo "Server failed to start within 90s"; \
		docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.auth.yml logs --tail=80; \
		docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.auth.yml down; \
		exit 1; \
	fi
	@rc=0; \
	uv run --package aegra-api pytest libs/aegra-api/tests/e2e/manual_auth_tests/ -v --tb=short -m auth_only -o addopts="--strict-markers -ra --color=yes" || rc=$$?; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.auth.yml down; \
	exit $$rc

e2e-both: e2e-dev e2e-prod

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ty_cache .ruff_cache htmlcov 2>/dev/null || true

run:
	uv run --package aegra-api uvicorn aegra_api.main:app --reload
