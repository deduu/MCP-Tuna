from __future__ import annotations

import json
import tomllib
from pathlib import Path


SERVER_NAME = "io.github.deduu/mcp-tuna"


def _load_server_manifest() -> dict:
    manifest_path = Path(__file__).resolve().parent.parent / "server.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_pyproject_version() -> str:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_server_manifest_uses_registry_namespace() -> None:
    manifest = _load_server_manifest()
    assert manifest["name"] == SERVER_NAME
    assert manifest["repository"]["url"] == "https://github.com/deduu/MCP-Tuna"
    assert manifest["repository"]["source"] == "github"


def test_server_manifest_pypi_package_entry_matches_publish_surface() -> None:
    manifest = _load_server_manifest()
    packages = manifest["packages"]
    assert len(packages) == 1

    package = packages[0]
    assert package["registry_type"] == "pypi"
    assert package["identifier"] == "mcp-tuna"
    assert manifest["version"] == _load_pyproject_version()
    assert package["version"] == _load_pyproject_version()
    assert package["transport"]["type"] == "stdio"
    assert package["runtime_hint"] == "uvx"
    assert package["runtime_arguments"] == [
        {"type": "named", "name": "--from", "value": "mcp-tuna[all-servers]"}
    ]
    assert package["package_arguments"] == [
        {"type": "positional", "value": "mcp-tuna-gateway"}
    ]


def test_readme_contains_mcp_registry_name_marker() -> None:
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    assert f"<!-- mcp-name: {SERVER_NAME} -->" in readme
