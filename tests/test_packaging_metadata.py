from __future__ import annotations

import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        return tomllib.load(f)


def test_wheel_includes_gateway_runtime_package() -> None:
    config = _load_pyproject()
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "app" in packages


def test_setup_cli_is_exposed() -> None:
    config = _load_pyproject()
    scripts = config["project"]["scripts"]
    assert scripts["mcp-tuna-setup"] == "scripts.setup_mcp:main"
