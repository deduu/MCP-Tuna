"""Package version helpers for installed and source-tree usage."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_package_version(
    dist_name: str = "mcp-tuna",
    fallback: str = "0.2.0",
) -> str:
    """Return the installed distribution version with a source fallback."""
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return fallback
