from __future__ import annotations

from pathlib import Path

from shared.owned_paths import resolve_owned_output_path
from shared.workspace_paths import get_workspace_root


def test_resolve_owned_output_path_scopes_relative_path_to_workspace_and_user():
    resolved = resolve_owned_output_path(
        "output/training/demo-run",
        {"workspace_id": "alpha-ws", "user_id": "user-1"},
    )

    expected = (
        get_workspace_root()
        / "output"
        / "workspaces"
        / "alpha-ws"
        / "users"
        / "user-1"
        / "training"
        / "demo-run"
    ).resolve()
    assert resolved == expected


def test_resolve_owned_output_path_preserves_absolute_paths(tmp_path: Path):
    absolute = tmp_path / "already-absolute"

    resolved = resolve_owned_output_path(
        str(absolute),
        {"workspace_id": "alpha-ws"},
    )

    assert resolved == absolute.resolve()
