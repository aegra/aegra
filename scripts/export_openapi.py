"""Export the OpenAPI spec to docs/openapi.json

Generates the spec from the FastAPI app, normalizes the title/version to
match the Aegra project defaults, and removes any custom-route endpoints
that are project-specific rather than part of the standard API.

NOTE: Environment variables must be set before importing the app, so the
imports below are intentionally placed after os.environ calls.
"""

import json
import logging
import os
from importlib.metadata import version
from pathlib import Path

# Suppress all logging before imports — env vars must be set first
logging.disable(logging.CRITICAL)
os.environ["AEGRA__APP__LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("AEGRA__APP__PROJECT_NAME", "Aegra")
os.environ.setdefault("AEGRA__APP__VERSION", version("aegra-api"))
os.environ.setdefault("AEGRA__APP__DEBUG", "false")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://export:export@localhost:5432/export")

import structlog  # noqa: E402

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
)

from aegra_api.main import OPENAPI_TAGS, app  # noqa: E402
from aegra_api.settings import settings  # noqa: E402


def _normalize_default_graph_id(schema: dict) -> None:
    """Publish ``graph_id`` as required, whatever config the exporter ran under.

    A deployment that resolves a default serves its own schema with ``graph_id``
    optional, which is correct for that server. This file is the canonical
    reference rather than any one deployment, so it states the contract every
    server honours — an explicit ``graph_id`` is always accepted — and does not
    change with the exporter's ``aegra.json``.
    """
    create = schema.get("components", {}).get("schemas", {}).get("AssistantCreate")
    if create is None:
        return

    graph_id = create["properties"]["graph_id"]
    graph_id.pop("default", None)
    graph_id.pop("anyOf", None)
    graph_id["type"] = "string"
    required = create.setdefault("required", [])
    if "graph_id" not in required:
        required.append("graph_id")


def main() -> None:
    """Export the OpenAPI spec to docs/openapi.json."""
    schema = app.openapi()

    # Normalize title and description to standard Aegra values
    schema["info"]["title"] = "Aegra"
    schema["info"]["version"] = settings.app.VERSION
    schema["info"]["description"] = "Production-ready Agent Protocol server"

    _normalize_default_graph_id(schema)

    # Remove custom-route endpoints and untagged paths (root, custom routes)
    core_tags: set[str] = {t["name"] for t in OPENAPI_TAGS}
    paths_to_remove: list[str] = []
    for path, methods in schema.get("paths", {}).items():
        for _method, info in methods.items():
            tags = set(info.get("tags", []))
            if not tags or not tags & core_tags:
                paths_to_remove.append(path)
                break

    for path in paths_to_remove:
        del schema["paths"][path]

    # Keep only core tags
    schema["tags"] = [t for t in schema.get("tags", []) if t["name"] in core_tags]

    output_path = Path(__file__).parent.parent / "docs" / "openapi.json"
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(schema['paths'])} endpoints to {output_path}")


if __name__ == "__main__":
    main()
