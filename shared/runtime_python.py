from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def resolve_repo_python(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def should_reexec_into_repo_python(
    *,
    repo_root: Path,
    current_executable: str,
    env_var: str,
) -> tuple[bool, Path | None]:
    preferred_python = resolve_repo_python(repo_root)
    if preferred_python is None:
        return False, None
    if os.getenv(env_var) == "1":
        return False, preferred_python
    try:
        current_path = Path(current_executable).resolve()
    except OSError:
        current_path = Path(current_executable)
    if current_path == preferred_python.resolve():
        return False, preferred_python
    return True, preferred_python


def build_reexec_command(
    preferred_python: Path,
    script_path: Path,
    argv: Sequence[str],
) -> list[str]:
    return [str(preferred_python), str(script_path), *list(argv)]
