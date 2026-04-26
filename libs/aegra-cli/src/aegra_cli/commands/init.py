"""Initialize a new Aegra project with interactive template selection."""

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template

import click
from rich.console import Console

from aegra_cli import __version__
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

console = Console()
logger = logging.getLogger(__name__)


def _is_interactive() -> bool:
    """Check if running in an interactive terminal.

    Returns:
        True if stdin is a TTY, False otherwise (CI, piped input, etc.).
    """
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        logger.debug("Failed to determine TTY status for stdin", exc_info=True)
        return False


def _resolve_name(path: Path, name: str | None) -> str:
    """Determine project name from explicit --name or from the path.

    Args:
        path: Resolved project directory.
        name: Explicit name from CLI flag, or None.

    Returns:
        Human-readable project name.
    """
    if name is not None:
        return name
    return path.name


def _prompt_path(default: str) -> str:
    """Interactively ask the user where to create the project.

    Args:
        default: Default path shown in the prompt.

    Returns:
        User-chosen path string.
    """
    value = click.prompt(
        click.style("\U0001f4c2 Where should I create the project?", bold=True),
        default=default,
    )
    return value


def _prompt_template(templates: Sequence[Mapping[str, str]]) -> int:
    """Interactively ask the user to pick a template.

    Args:
        templates: List of template dicts with name and description.

    Returns:
        1-based template index.
    """
    click.echo()
    click.echo(click.style("\U0001f31f Choose a template:", bold=True))
    for i, t in enumerate(templates, 1):
        click.echo(f"  {i}. {t['name']} \u2014 {t['description']}")
    click.echo()
    choice = click.prompt("Enter your choice", default=1, type=int)
    return choice


def _write_file(path: Path, content: str, force: bool) -> bool:
    """Write content to a file, respecting the force flag.

    Args:
        path: Path to write to.
        content: Content to write.
        force: Whether to overwrite existing files.

    Returns:
        True if file was written, False if skipped.
    """
    if path.exists() and not force:
        console.print(f"  [yellow]SKIP[/yellow]    {path}")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    console.print(f"  [green]CREATE[/green]  {path}")
    return True


def _resolve_template_by_name(query: str, templates: Sequence[Mapping[str, str]]) -> int:
    """Match a template by id or name substring.

    Returns:
        1-based template index, or -1 if no match found.
    """
    q = query.lower()
    for i, t in enumerate(templates, 1):
        if q == t["id"].lower():
            return i
    # Fallback: substring match on id or name
    for i, t in enumerate(templates, 1):
        if q in t["id"].lower() or q in t["name"].lower():
            return i
    console.print(
        f"[bold red]Error:[/bold red] Unknown template '{query}'. "
        f"Available: {', '.join(t['id'] for t in templates)}"
    )
    return -1


@click.command()
@click.argument("path", default=".", required=False)
@click.option(
    "--template",
    "-t",
    type=str,
    default=None,
    help="Template number or name (e.g. 1, langgraphjs, react).",
)
@click.option("--name", "-n", default=None, help="Project name (defaults to directory name).")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def init(path: str, template: str | None, name: str | None, force: bool) -> None:
    """Initialize a new Aegra project.

    Creates a complete project structure from a template, including:

    \b
    - aegra.json: Graph configuration
    - pyproject.toml: Project dependencies
    - .env.example: Environment variable template
    - .gitignore: Standard Python gitignore
    - README.md: Project readme
    - src/<slug>/__init__.py: Package init
    - src/<slug>/graph.py: Template-specific graph
    - src/<slug>/state.py, prompts.py, context.py, utils.py: Shared modules
    - src/<slug>/tools.py: Tool definitions (ReAct agent only)
    - docker-compose.yml: Docker Compose (PostgreSQL + API)
    - Dockerfile: Production container build

    Examples:

    \b
        aegra init                           # Interactive mode
        aegra init ./my-agent                # Create at path
        aegra init ./my-agent -t 1           # New Aegra Project
        aegra init ./my-agent -t react       # ReAct Agent (by name)
        aegra init ./my-agent -t langgraphjs # LangGraph.js Chatbot
        aegra init ./my-agent -t 1 -n "My Agent"
    """
    templates = get_template_choices()
    interactive = _is_interactive()

    # --- Resolve path (interactive if default ".") ---
    # Prompt for path when using default "." in interactive mode,
    # unless both --template and --name are given (full automation).
    if path == "." and interactive and not (template and name):
        path = _prompt_path(".")
    project_path = Path(path).resolve()

    # --- Resolve template ---
    if template is None:
        if interactive:
            choice = _prompt_template(templates)
        else:
            # Non-interactive: default to template 1
            choice = 1
    else:
        # Try numeric index first, then match by id/name substring
        try:
            choice = int(template)
        except ValueError:
            choice = _resolve_template_by_name(template, templates)

    if choice < 1 or choice > len(templates):
        console.print(f"[bold red]Error:[/bold red] Invalid template number: {choice}")
        raise SystemExit(1)

    selected = templates[choice - 1]
    click.echo(f"You selected: {selected['name']} \u2014 {selected['description']}")

    # --- Resolve project name ---
    project_name = _resolve_name(project_path, name)
    slug = slugify(project_name)

    # Template variables
    variables = {
        "project_name": project_name,
        "slug": slug,
        "aegra_version": __version__,
    }

    click.echo()
    click.echo(click.style("\U0001f4e5 Creating project...", bold=True))

    files_created = 0
    files_skipped = 0

    def _write(rel_path: str, content: str) -> None:
        nonlocal files_created, files_skipped
        full = project_path / rel_path
        if _write_file(full, content, force):
            files_created += 1
        else:
            files_skipped += 1

    # --- aegra.json ---
    manifest = load_template_manifest(selected["id"])
    is_js_template = manifest.get("runtime") == "langgraphjs"

    if is_js_template:
        # LangGraph.js templates use the runtime object format
        aegra_config = {
            "name": project_name,
            "dependencies": ["./src"],
            "graphs": {
                slug: {
                    "runtime": "langgraphjs",
                    "path": f"./src/{slug}/graph.ts:graph",
                }
            },
        }
    else:
        aegra_config = {
            "name": project_name,
            "dependencies": ["./src"],
            "graphs": {slug: f"./src/{slug}/graph.py:graph"},
        }
    _write("aegra.json", json.dumps(aegra_config, indent=2) + "\n")

    # --- Template files (from manifest) ---
    manifest = load_template_manifest(selected["id"])
    for template_filename, dest_pattern in manifest["files"].items():
        dest = Template(dest_pattern).safe_substitute(variables)
        content = render_template_file(selected["id"], template_filename, variables)
        _write(dest, content)

    # --- Shared template files (from manifest) ---
    for shared_filename, dest_pattern in manifest.get("shared_files", {}).items():
        dest = Template(dest_pattern).safe_substitute(variables)
        content = render_shared_template_file(shared_filename, variables)
        _write(dest, content)

    # --- .env.example ---
    _write(".env.example", render_env_example(variables))

    # --- .gitignore ---
    _write(".gitignore", load_shared_file("gitignore"))

    # --- Docker files ---
    _write("docker-compose.yml", get_docker_compose(slug))
    _write("Dockerfile", get_dockerfile(runtime="langgraphjs" if is_js_template else "python"))

    # --- Summary ---
    click.echo()
    click.echo(f"\U0001f389 New project created at {project_path}")
    click.echo()

    if files_skipped:
        console.print(
            f"[green]{files_created}[/green] files created, "
            f"[yellow]{files_skipped}[/yellow] files skipped"
        )
        click.echo()

    click.echo(click.style("Next steps:", bold=True))
    click.echo(f"  cd {project_path.name}")
    click.echo("  cp .env.example .env       # Configure your environment")
    if is_js_template:
        click.echo("  npm install                # Install Node.js dependencies")
        click.echo("  pip install aegra-cli      # Install Aegra CLI (if needed)")
    else:
        click.echo("  uv sync                    # Install dependencies")
    click.echo("  uv run aegra dev           # Start developing!")
