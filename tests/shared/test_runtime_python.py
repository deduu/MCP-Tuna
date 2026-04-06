from __future__ import annotations

from pathlib import Path

from shared.runtime_python import (
    build_reexec_command,
    resolve_repo_python,
    should_reexec_into_repo_python,
)


def test_resolve_repo_python_prefers_windows_venv_path(tmp_path):
    python_path = tmp_path / ".venv" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")

    assert resolve_repo_python(tmp_path) == python_path


def test_should_reexec_into_repo_python_when_current_differs(tmp_path, monkeypatch):
    python_path = tmp_path / ".venv" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("WA_PREF_COMPARE_REEXECED", raising=False)

    should_reexec, preferred = should_reexec_into_repo_python(
        repo_root=tmp_path,
        current_executable=str(tmp_path / "python.exe"),
        env_var="WA_PREF_COMPARE_REEXECED",
    )

    assert should_reexec is True
    assert preferred == python_path


def test_should_not_reexec_when_already_using_repo_python(tmp_path, monkeypatch):
    python_path = tmp_path / ".venv" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("WA_PREF_COMPARE_REEXECED", raising=False)

    should_reexec, preferred = should_reexec_into_repo_python(
        repo_root=tmp_path,
        current_executable=str(python_path),
        env_var="WA_PREF_COMPARE_REEXECED",
    )

    assert should_reexec is False
    assert preferred == python_path


def test_should_not_reexec_when_guard_env_is_set(tmp_path, monkeypatch):
    python_path = tmp_path / ".venv" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("WA_PREF_COMPARE_REEXECED", "1")

    should_reexec, preferred = should_reexec_into_repo_python(
        repo_root=tmp_path,
        current_executable=str(tmp_path / "python.exe"),
        env_var="WA_PREF_COMPARE_REEXECED",
    )

    assert should_reexec is False
    assert preferred == python_path


def test_build_reexec_command_preserves_script_and_args():
    preferred_python = Path("C:/repo/.venv/Scripts/python.exe")
    script_path = Path("C:/repo/scripts/compare_wa_sales_preference.py")

    command = build_reexec_command(preferred_python, script_path, ["--foo", "bar"])

    assert command == [
        str(preferred_python),
        str(script_path),
        "--foo",
        "bar",
    ]
