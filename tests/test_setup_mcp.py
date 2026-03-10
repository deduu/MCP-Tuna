from __future__ import annotations

from pathlib import Path

from scripts.setup_mcp import (
    DEFAULT_PACKAGE_SPEC,
    build_stdio_entry,
    generate_claude_desktop_config,
    generate_vscode_config,
)


def test_build_stdio_entry_installed_launcher() -> None:
    entry = build_stdio_entry(launcher="installed")
    assert entry == {"command": "mcp-tuna-gateway"}


def test_build_stdio_entry_uvx_launcher() -> None:
    entry = build_stdio_entry(
        launcher="uvx",
        package_spec=DEFAULT_PACKAGE_SPEC,
    )
    assert entry == {
        "command": "uvx",
        "args": ["--from", DEFAULT_PACKAGE_SPEC, "mcp-tuna-gateway"],
    }


def test_generate_vscode_config_defaults_to_installed_console_script() -> None:
    config = generate_vscode_config(
        Path.cwd(),
        transport="stdio",
        port=8002,
        launcher="installed",
        package_spec=DEFAULT_PACKAGE_SPEC,
    )
    assert config == {
        "servers": {
            "mcp-tuna-gateway": {
                "command": "mcp-tuna-gateway",
            }
        }
    }


def test_generate_claude_desktop_config_supports_repo_launcher() -> None:
    config = generate_claude_desktop_config(
        Path("C:/repo"),
        transport="stdio",
        port=8002,
        launcher="repo",
        package_spec=DEFAULT_PACKAGE_SPEC,
    )
    assert config == {
        "mcpServers": {
            "mcp-tuna-gateway": {
                "command": "uv",
                "args": [
                    "--directory",
                    "C:\\repo",
                    "run",
                    "python",
                    "-m",
                    "scripts.run_gateway",
                ],
            }
        }
    }
