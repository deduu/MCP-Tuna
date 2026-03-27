from __future__ import annotations

from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def get_workspace_root() -> Path:
    return WORKSPACE_ROOT


def resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (get_workspace_root() / candidate).resolve()


def to_workspace_relative_path(path: str | Path) -> str:
    resolved = resolve_workspace_path(path)
    try:
        relative = resolved.relative_to(get_workspace_root())
        return str(relative).replace("\\", "/")
    except ValueError:
        return resolved.name
