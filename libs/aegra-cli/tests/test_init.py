"""Tests for the init command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

import aegra_cli.commands.init  # noqa: F401 — ensure module is loaded
from aegra_cli.cli import cli
from aegra_cli.templates import (
    get_docker_compose,
    get_dockerfile,
    get_template_choices,
    load_shared_file,
    load_template_manifest,
    render_env_example,
    render_shared_template_file,
    render_template_file,
    slugify,
)

if TYPE_CHECKING:
    from pytest import MonkeyPatch


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    """Tests for the slugify function."""

    def test_simple_name(self: TestSlugify) -> None:
        assert slugify("myproject") == "myproject"

    def test_with_spaces(self: TestSlugify) -> None:
        assert slugify("My Project") == "my_project"

    def test_with_hyphens(self: TestSlugify) -> None:
        assert slugify("my-project") == "my_project"

    def test_with_special_chars(self: TestSlugify) -> None:
        assert slugify("My App 2.0!") == "my_app_20"

    def test_with_leading_number(self: TestSlugify) -> None:
        assert slugify("123project") == "project_123project"

    def test_empty_string(self: TestSlugify) -> None:
        assert slugify("") == "aegra_project"

    def test_only_special_chars(self: TestSlugify) -> None:
        assert slugify("!@#$%") == "aegra_project"


# ---------------------------------------------------------------------------
# Template registry & renderers
# ---------------------------------------------------------------------------


class TestTemplateRegistry:
    """Tests for template registry helpers."""

    def test_get_template_choices_returns_list(self: TestTemplateRegistry) -> None:
        choices = get_template_choices()
        assert isinstance(choices, list)
        assert len(choices) >= 2

    def test_each_template_has_required_keys(self: TestTemplateRegistry) -> None:
        for t in get_template_choices():
            assert "id" in t
            assert "name" in t
            assert "description" in t

    def test_load_manifest_simple_chatbot(self: TestTemplateRegistry) -> None:
        manifest = load_template_manifest("simple-chatbot")
        assert "files" in manifest
        assert "graph.py.template" in manifest["files"]

    def test_load_manifest_react_agent(self: TestTemplateRegistry) -> None:
        manifest = load_template_manifest("react-agent")
        assert "files" in manifest
        assert "tools.py.template" in manifest["files"]

    def test_manifest_has_shared_files(self: TestTemplateRegistry) -> None:
        for template_id in ("simple-chatbot", "react-agent"):
            manifest = load_template_manifest(template_id)
            assert "shared_files" in manifest
            shared = manifest["shared_files"]
            assert "state.py.template" in shared
            assert "prompts.py.template" in shared
            assert "context.py.template" in shared
            assert "utils.py.template" in shared

    def test_render_template_file_substitutes_variables(self: TestTemplateRegistry) -> None:
        content = render_template_file(
            "simple-chatbot",
            "graph.py.template",
            {"project_name": "My Bot", "slug": "my_bot"},
        )
        assert "My Bot" in content
        assert "$project_name" not in content

    def test_render_shared_template_file_substitutes_variables(self: TestTemplateRegistry) -> None:
        content = render_shared_template_file(
            "state.py.template",
            {"project_name": "My Bot", "slug": "my_bot"},
        )
        assert "My Bot" in content
        assert "$project_name" not in content

    def test_render_env_example_substitutes_slug(self: TestTemplateRegistry) -> None:
        content = render_env_example({"slug": "test_app"})
        assert "test_app" in content
        assert "POSTGRES_USER" in content

    def test_load_shared_gitignore(self: TestTemplateRegistry) -> None:
        content = load_shared_file("gitignore")
        assert "__pycache__" in content
        assert ".env" in content


# ---------------------------------------------------------------------------
# Docker generators
# ---------------------------------------------------------------------------


class TestDockerGenerators:
    """Tests for Docker file generators."""

    def test_docker_compose_has_postgres(self: TestDockerGenerators) -> None:
        compose = get_docker_compose("myapp")
        assert "postgres:" in compose
        assert "myapp-postgres" in compose

    def test_docker_compose_has_api_service(self: TestDockerGenerators) -> None:
        compose = get_docker_compose("myapp")
        assert "myapp:" in compose
        assert "myapp-api" in compose
        assert "build:" in compose

    def test_docker_compose_mounts_src(self: TestDockerGenerators) -> None:
        compose = get_docker_compose("myapp")
        assert "./src:/app/src:ro" in compose

    def test_docker_compose_has_api_healthcheck(self: TestDockerGenerators) -> None:
        compose = get_docker_compose("myapp")
        assert "curl -sf http://localhost:${PORT:-2026}/health || exit 1" in compose

    def test_docker_compose_has_restart_policy(self: TestDockerGenerators) -> None:
        compose = get_docker_compose("myapp")
        assert compose.count("restart: unless-stopped") == 3  # postgres + redis + api

    def test_docker_compose_has_redis(self: TestDockerGenerators) -> None:
        compose = get_docker_compose("myapp")
        assert "redis:" in compose
        assert "myapp-redis" in compose
        assert "redis:7-alpine" in compose
        assert "REDIS_BROKER_ENABLED=true" in compose
        assert "REDIS_URL=redis://redis:6379/0" in compose

    def test_docker_compose_api_depends_on_postgres_and_redis(self: TestDockerGenerators) -> None:
        compose = get_docker_compose("myapp")
        # Extract the api service's depends_on block by finding the section
        # between "depends_on:" and the next 4-space-indented key
        lines = compose.splitlines()
        in_api_service = False
        in_depends_on = False
        depends_on_lines: list[str] = []
        for line in lines:
            stripped = line.rstrip()
            # Detect api service block (container_name has the slug)
            if "container_name: myapp-api" in stripped:
                in_api_service = True
            # Exit api service block on next top-level service (2-space indent with colon)
            elif in_api_service and stripped and not stripped.startswith(" "):
                in_api_service = False
                in_depends_on = False
            if in_api_service and stripped.strip() == "depends_on:":
                in_depends_on = True
                continue
            if in_depends_on:
                # depends_on entries are at 6+ spaces; a line at 4 spaces exits the block
                if stripped and not stripped.startswith("      "):
                    in_depends_on = False
                else:
                    depends_on_lines.append(stripped)
        depends_on_block = "\n".join(depends_on_lines)
        assert "postgres:" in depends_on_block, "API service should depend on postgres"
        assert "redis:" in depends_on_block, "API service should depend on redis"
        assert "service_healthy" in depends_on_block

    def test_dockerfile_installs_project(self: TestDockerGenerators) -> None:
        dockerfile = get_dockerfile()
        assert "FROM python" in dockerfile
        assert "uv sync" in dockerfile
        assert "COPY pyproject.toml" in dockerfile
        assert "COPY src/" in dockerfile
        assert "EXPOSE 2026" in dockerfile

    def test_dockerfile_copies_src_in_final_stage(self: TestDockerGenerators) -> None:
        """src/ must be copied in the runtime stage, not just the builder.

        Without this, cloud deployments (Railway, Render, Fly.io) fail because
        aegra.json file-path references (graphs, http.app, auth.path) resolve
        against the filesystem.  Volume mounts in docker-compose mask this
        during local dev.  See #255.
        """
        dockerfile = get_dockerfile()
        # Split at the final stage marker — fail fast if marker is missing
        parts = dockerfile.split("FROM base AS final")
        assert len(parts) == 2, "Expected exactly one 'FROM base AS final' stage marker"
        final_stage = parts[1]
        assert "COPY src/ ./src/" in final_stage

    def test_dockerfile_security_and_best_practices(self: TestDockerGenerators) -> None:
        dockerfile = get_dockerfile()
        # Non-root user
        assert "addgroup --system app" in dockerfile
        assert "USER app" in dockerfile
        # Pinned uv version (not :latest)
        assert "uv:0." in dockerfile
        assert "uv:latest" not in dockerfile
        # uv.lock optionally copied (glob pattern for optional file)
        assert "uv.loc[k]" in dockerfile
        # Runtime essentials
        assert "PYTHONUNBUFFERED=1" in dockerfile
        assert "ca-certificates" in dockerfile
        assert "libpq5" in dockerfile


# ---------------------------------------------------------------------------
# init command — interactive (CliRunner input=)
# ---------------------------------------------------------------------------


class TestInitInteractive:
    """Tests for interactive init prompts."""

    def test_interactive_default_path_template_1(
        self: TestInitInteractive,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Interactive flow: default path, pick template 1."""
        monkeypatch.setattr(sys.modules["aegra_cli.commands.init"], "_is_interactive", lambda: True)
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["init"], input=".\n1\n")
            assert result.exit_code == 0
            assert Path("aegra.json").exists()
            assert Path("pyproject.toml").exists()

    def test_interactive_custom_path(
        self: TestInitInteractive,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Interactive flow: enter a custom directory."""
        monkeypatch.setattr(sys.modules["aegra_cli.commands.init"], "_is_interactive", lambda: True)
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["init"], input="./my-agent\n1\n")
            assert result.exit_code == 0
            assert Path("my-agent/aegra.json").exists()
            assert Path("my-agent/pyproject.toml").exists()

    def test_interactive_template_2(
        self: TestInitInteractive,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Interactive flow: pick template 2 (ReAct agent)."""
        monkeypatch.setattr(sys.modules["aegra_cli.commands.init"], "_is_interactive", lambda: True)
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["init"], input=".\n2\n")
            assert result.exit_code == 0
            # ReAct agent should have tools.py
            slug = slugify(Path.cwd().name)
            assert Path(f"src/{slug}/tools.py").exists()


# ---------------------------------------------------------------------------
# init command — CLI flags (non-interactive)
# ---------------------------------------------------------------------------


class TestInitCLIFlags:
    """Tests for non-interactive CLI flag usage."""

    def test_path_argument_and_template_flag(
        self: TestInitCLIFlags, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """aegra init ./my-agent -t 1"""
        project_dir = tmp_path / "my-agent"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0
        assert (project_dir / "aegra.json").exists()
        assert (project_dir / "pyproject.toml").exists()
        assert (project_dir / ".env.example").exists()
        assert (project_dir / ".gitignore").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "docker-compose.yml").exists()
        assert (project_dir / "Dockerfile").exists()
        # No docker-compose.prod.yml
        assert not (project_dir / "docker-compose.prod.yml").exists()

    def test_shared_files_created(
        self: TestInitCLIFlags, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """aegra init creates shared template files (state, prompts, context, utils)."""
        project_dir = tmp_path / "shared-test"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("shared-test")
        assert (project_dir / f"src/{slug}/state.py").exists()
        assert (project_dir / f"src/{slug}/prompts.py").exists()
        assert (project_dir / f"src/{slug}/context.py").exists()
        assert (project_dir / f"src/{slug}/utils.py").exists()

    def test_path_with_name_flag(
        self: TestInitCLIFlags, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """aegra init ./my-agent -t 1 -n 'My Agent'"""
        project_dir = tmp_path / "my-agent"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1", "-n", "My Agent"])
        assert result.exit_code == 0

        config = json.loads((project_dir / "aegra.json").read_text(encoding="utf-8"))
        assert config["name"] == "My Agent"
        assert "my_agent" in config["graphs"]

    def test_react_template_creates_tools(
        self: TestInitCLIFlags, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """aegra init -t 2 creates tools.py for ReAct agent."""
        project_dir = tmp_path / "react-test"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "2"])
        assert result.exit_code == 0

        slug = slugify("react-test")
        assert (project_dir / f"src/{slug}/tools.py").exists()
        assert (project_dir / f"src/{slug}/graph.py").exists()

        # tools.py should contain tool definitions
        tools_content = (project_dir / f"src/{slug}/tools.py").read_text(encoding="utf-8")
        assert "TOOLS" in tools_content
        assert "@tool" in tools_content

    def test_name_from_path(self: TestInitCLIFlags, cli_runner: CliRunner, tmp_path: Path) -> None:
        """When no --name, project name derives from directory."""
        project_dir = tmp_path / "my-cool-agent"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        config = json.loads((project_dir / "aegra.json").read_text(encoding="utf-8"))
        assert config["name"] == "my-cool-agent"

    def test_invalid_template_number(
        self: TestInitCLIFlags, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Invalid template number shows error."""
        project_dir = tmp_path / "bad-template"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "99"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# File content validation
# ---------------------------------------------------------------------------


class TestInitFileContents:
    """Tests for the content of generated files."""

    def test_aegra_json_has_dependencies(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-deps"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        config = json.loads((project_dir / "aegra.json").read_text(encoding="utf-8"))
        assert "dependencies" in config
        assert "./src" in config["dependencies"]

    def test_aegra_json_graph_path_uses_src(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-graph"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        config = json.loads((project_dir / "aegra.json").read_text(encoding="utf-8"))
        slug = slugify("test-graph")
        assert config["graphs"][slug] == f"./src/{slug}/graph.py:graph"

    def test_pyproject_toml_has_aegra_dep(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-pyproject"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        content = (project_dir / "pyproject.toml").read_text(encoding="utf-8")
        assert "aegra-cli" in content
        assert "langgraph" in content
        assert "langchain-openai" in content
        assert "langchain-anthropic" in content
        assert "langchain>=" in content

    def test_graph_has_langgraph_imports(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-graph-imports"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("test-graph-imports")
        content = (project_dir / f"src/{slug}/graph.py").read_text(encoding="utf-8")
        assert "from langgraph.graph import" in content
        assert "StateGraph" in content

    def test_graph_uses_state_and_context(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-state-ctx"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("test-state-ctx")
        content = (project_dir / f"src/{slug}/graph.py").read_text(encoding="utf-8")
        assert f"from {slug}.context import Context" in content
        assert f"from {slug}.state import InputState, State" in content
        assert f"from {slug}.utils import load_chat_model" in content

    def test_graph_exports_graph_variable(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-graph-var"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("test-graph-var")
        content = (project_dir / f"src/{slug}/graph.py").read_text(encoding="utf-8")
        assert "graph =" in content or "graph:" in content

    def test_env_example_has_required_vars(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-env"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        content = (project_dir / ".env.example").read_text(encoding="utf-8")
        for var in ["POSTGRES_USER", "POSTGRES_PASSWORD", "AUTH_TYPE", "OPENAI_API_KEY"]:
            assert var in content

    def test_env_example_uses_slug(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "slug-test"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1", "-n", "My App"])
        assert result.exit_code == 0

        content = (project_dir / ".env.example").read_text(encoding="utf-8")
        assert "my_app" in content

    def test_gitignore_has_standard_entries(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-gitignore"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        content = (project_dir / ".gitignore").read_text(encoding="utf-8")
        assert "__pycache__" in content
        assert ".env" in content
        assert ".venv" in content

    def test_readme_has_project_name(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-readme"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1", "-n", "Cool Agent"])
        assert result.exit_code == 0

        content = (project_dir / "README.md").read_text(encoding="utf-8")
        assert "Cool Agent" in content

    def test_docker_compose_has_both_services(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-compose"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        content = (project_dir / "docker-compose.yml").read_text(encoding="utf-8")
        assert "postgres:" in content
        assert "build:" in content
        slug = slugify("test-compose")
        assert f"{slug}:" in content

    def test_docker_compose_has_api_healthcheck(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-healthcheck"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        content = (project_dir / "docker-compose.yml").read_text(encoding="utf-8")
        assert "curl -sf http://localhost:${PORT:-2026}/health || exit 1" in content

    def test_no_prod_compose_generated(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-no-prod"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0
        assert not (project_dir / "docker-compose.prod.yml").exists()

    def test_dockerfile_installs_from_pyproject(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-docker"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        content = (project_dir / "Dockerfile").read_text(encoding="utf-8")
        assert "COPY pyproject.toml" in content
        assert "COPY src/" in content
        assert "uv sync" in content

    def test_state_py_has_input_and_state_classes(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-state"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("test-state")
        content = (project_dir / f"src/{slug}/state.py").read_text(encoding="utf-8")
        assert "class InputState" in content
        assert "class State(InputState)" in content
        assert "is_last_step" in content

    def test_context_py_has_context_class(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-context"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("test-context")
        content = (project_dir / f"src/{slug}/context.py").read_text(encoding="utf-8")
        assert "class Context" in content
        assert "system_prompt" in content
        assert "model" in content
        assert "openai/gpt-4o-mini" in content

    def test_utils_py_has_load_chat_model(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-utils"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("test-utils")
        content = (project_dir / f"src/{slug}/utils.py").read_text(encoding="utf-8")
        assert "def load_chat_model" in content
        assert "init_chat_model" in content

    def test_prompts_py_has_system_prompt(
        self: TestInitFileContents, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "test-prompts"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("test-prompts")
        content = (project_dir / f"src/{slug}/prompts.py").read_text(encoding="utf-8")
        assert "SYSTEM_PROMPT" in content
        assert "system_time" in content


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestInitEdgeCases:
    """Tests for edge cases in init command."""

    def test_force_overwrites_existing_files(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "force-test"
        project_dir.mkdir()
        (project_dir / "aegra.json").write_text('{"old": true}')

        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1", "--force"])
        assert result.exit_code == 0

        config = json.loads((project_dir / "aegra.json").read_text(encoding="utf-8"))
        assert "graphs" in config
        assert "old" not in config

    def test_skips_existing_files_without_force(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "skip-test"
        project_dir.mkdir()
        (project_dir / "aegra.json").write_text('{"old": true}')

        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0
        assert "SKIP" in result.output

        config = json.loads((project_dir / "aegra.json").read_text(encoding="utf-8"))
        assert config == {"old": True}

    def test_creates_nested_directories(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "deep" / "nested" / "project"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0
        assert (project_dir / "aegra.json").exists()

    def test_init_in_nonempty_directory(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "nonempty"
        project_dir.mkdir()
        (project_dir / "existing.txt").write_text("keep me")

        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0
        assert (project_dir / "existing.txt").read_text(encoding="utf-8") == "keep me"
        assert (project_dir / "aegra.json").exists()

    def test_double_init_without_force_skips(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "double"
        cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0
        assert "SKIP" in result.output

    def test_double_init_with_force_overwrites(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "double-force"
        cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1", "--force"])
        assert result.exit_code == 0
        assert "CREATE" in result.output

    def test_shows_next_steps(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "steps-test"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0
        assert "Next steps" in result.output
        assert "aegra dev" in result.output
        assert "uv run aegra dev" in result.output

    def test_help_shows_options(self: TestInitEdgeCases, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "--template" in result.output
        assert "-t" in result.output
        assert "--name" in result.output
        assert "-n" in result.output
        assert "--force" in result.output

    def test_react_graph_imports_from_slug(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """ReAct graph.py should import tools from the correct package."""
        project_dir = tmp_path / "react-import"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "2"])
        assert result.exit_code == 0

        slug = slugify("react-import")
        content = (project_dir / f"src/{slug}/graph.py").read_text(encoding="utf-8")
        assert f"from {slug}.tools import TOOLS" in content

    def test_init_current_dir_with_template_flag(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """aegra init . -t 1 should work without interactive prompts."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["init", ".", "-t", "1"])
            assert result.exit_code == 0
            assert Path("aegra.json").exists()

    def test_project_name_substituted_in_graph(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "name-sub"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1", "-n", "Super Bot"])
        assert result.exit_code == 0

        slug = slugify("Super Bot")
        content = (project_dir / f"src/{slug}/graph.py").read_text(encoding="utf-8")
        assert "Super Bot" in content

    def test_react_shared_files_created(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """ReAct template also creates shared files."""
        project_dir = tmp_path / "react-shared"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "2"])
        assert result.exit_code == 0

        slug = slugify("react-shared")
        assert (project_dir / f"src/{slug}/state.py").exists()
        assert (project_dir / f"src/{slug}/prompts.py").exists()
        assert (project_dir / f"src/{slug}/context.py").exists()
        assert (project_dir / f"src/{slug}/utils.py").exists()
        assert (project_dir / f"src/{slug}/tools.py").exists()
        assert (project_dir / f"src/{slug}/graph.py").exists()

    def test_react_graph_has_route_model_output(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """ReAct graph should use custom route_model_output instead of tools_condition."""
        project_dir = tmp_path / "react-route"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "2"])
        assert result.exit_code == 0

        slug = slugify("react-route")
        content = (project_dir / f"src/{slug}/graph.py").read_text(encoding="utf-8")
        assert "route_model_output" in content
        assert "is_last_step" in content

    def test_context_py_imports_from_slug(
        self: TestInitEdgeCases, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """context.py should import prompts from the correct package."""
        project_dir = tmp_path / "ctx-import"
        result = cli_runner.invoke(cli, ["init", str(project_dir), "-t", "1"])
        assert result.exit_code == 0

        slug = slugify("ctx-import")
        content = (project_dir / f"src/{slug}/context.py").read_text(encoding="utf-8")
        assert f"from {slug} import prompts" in content
