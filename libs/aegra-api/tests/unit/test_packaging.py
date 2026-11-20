"""Packaging guarantees for code that imports aegra_api."""

from importlib.resources import files


def test_package_ships_pep_561_py_typed_marker() -> None:
    assert files("aegra_api").joinpath("py.typed").is_file()
