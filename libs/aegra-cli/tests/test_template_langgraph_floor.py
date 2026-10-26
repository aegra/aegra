"""New projects start on the same langgraph floor that aegra-api is tested against."""

import tomllib
from importlib.metadata import requires

import pytest
from packaging.requirements import Requirement

from aegra_cli.templates import get_template_choices, render_template_file


def _langgraph_specifier(requirements: list[str]) -> str:
    matches = [Requirement(r) for r in requirements if Requirement(r).name == "langgraph"]
    assert len(matches) == 1, f"expected one langgraph requirement, got {matches}"
    return str(matches[0].specifier)


@pytest.mark.parametrize("template_id", [t["id"] for t in get_template_choices()])
def test_template_langgraph_floor_matches_aegra_api(template_id: str) -> None:
    rendered = render_template_file(
        template_id, "pyproject.toml.template", {"aegra_version": "0.0.0"}
    )
    template_deps = tomllib.loads(rendered)["project"]["dependencies"]

    template_floor = _langgraph_specifier(template_deps)
    api_floor = _langgraph_specifier(requires("aegra-api") or [])

    assert template_floor == api_floor
