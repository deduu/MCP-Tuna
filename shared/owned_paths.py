from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.ownership import effective_ownership_context
from shared.workspace_paths import get_workspace_root


def _normalize_relative_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    parts = list(path.parts)
    if parts and parts[0] == "output":
        parts = parts[1:]
    return Path(*parts) if parts else Path()


def resolve_owned_output_path(path: str | Path, ownership: Any = None) -> Path:
    """Resolve training output paths under an owner-scoped workspace root.

    Absolute paths are preserved. Relative paths are rewritten under:

    - `output/workspaces/<workspace_id>/...`
    - `output/workspaces/<workspace_id>/users/<user_id>/...` when user_id exists
    """

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    normalized = _normalize_relative_output_path(candidate)
    ctx = effective_ownership_context(ownership)
    root = get_workspace_root() / "output" / "workspaces" / ctx.workspace_id
    if ctx.user_id:
        root = root / "users" / ctx.user_id
    return (root / normalized).resolve()
