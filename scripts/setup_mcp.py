"""Auto-generate MCP client configuration files for MCP Tuna.

Supports VS Code, Claude Desktop, Claude Code, and Cursor.

Usage:
    mcp-tuna-setup --all
    mcp-tuna-setup --vscode
    mcp-tuna-setup --transport http
    mcp-tuna-setup --launcher uvx
    mcp-tuna-setup --launcher repo
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

SERVER_NAME = "mcp-tuna-gateway"
DEFAULT_PACKAGE_SPEC = "mcp-tuna[all-servers]"


def detect_project_root() -> Path:
    """Walk up from this script's directory to find the repo root."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "mcp_gateway.py").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    print(
        "Error: Could not find project root (directory containing mcp_gateway.py).",
        file=sys.stderr,
    )
    sys.exit(1)


def resolve_project_root(launcher: str) -> Path:
    """Return the target directory for project-scoped MCP configs."""
    if launcher == "repo":
        return detect_project_root()
    return Path.cwd().resolve()


def build_stdio_entry(
    *,
    launcher: str,
    project_root: Optional[Path] = None,
    package_spec: str = DEFAULT_PACKAGE_SPEC,
    needs_abs_path: bool = False,
) -> dict:
    """Build a stdio-mode server entry dict."""
    if launcher == "installed":
        return {"command": "mcp-tuna-gateway"}

    if launcher == "uvx":
        return {
            "command": "uvx",
            "args": ["--from", package_spec, "mcp-tuna-gateway"],
        }

    if launcher == "repo":
        if project_root is None:
            raise ValueError("project_root is required when launcher='repo'")
        if needs_abs_path:
            return {
                "command": "uv",
                "args": [
                    "--directory",
                    str(project_root),
                    "run",
                    "python",
                    "-m",
                    "scripts.run_gateway",
                ],
            }
        return {
            "command": "uv",
            "args": ["run", "python", "-m", "scripts.run_gateway"],
        }

    raise ValueError(f"Unknown launcher: {launcher}")


def build_http_entry(port: int = 8002) -> dict:
    """Build an HTTP-mode server entry dict."""
    return {
        "type": "http",
        "url": f"http://localhost:{port}/mcp",
    }


def generate_vscode_config(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(
            launcher=launcher,
            project_root=project_root,
            package_spec=package_spec,
        )
    return {"servers": {SERVER_NAME: entry}}


def generate_claude_desktop_config(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(
            launcher=launcher,
            project_root=project_root,
            package_spec=package_spec,
            needs_abs_path=True,
        )
    return {"mcpServers": {SERVER_NAME: entry}}


def generate_claude_code_config(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(
            launcher=launcher,
            project_root=project_root,
            package_spec=package_spec,
        )
    return {"mcpServers": {SERVER_NAME: entry}}


def generate_cursor_config(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(
            launcher=launcher,
            project_root=project_root,
            package_spec=package_spec,
        )
    return {"mcpServers": {SERVER_NAME: entry}}


def safe_merge_config(
    path: Path,
    new_config: dict,
    server_key: str,
    server_name: str,
) -> bool:
    """Read existing config, back it up, merge only our server entry, and write."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            existing = {}
        else:
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"Error: {path} contains invalid JSON ({exc}). "
                    "Fix it manually or delete it, then re-run.",
                    file=sys.stderr,
                )
                return False

        backup = path.with_suffix(path.suffix + ".backup")
        shutil.copy2(path, backup)
        print(f"  Backed up existing config to {backup}")

        if server_key not in existing:
            existing[server_key] = {}
        existing[server_key][server_name] = new_config[server_key][server_name]
        merged = existing
    else:
        merged = new_config

    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {path}")
    return True


def _claude_desktop_config_path() -> Path:
    """Return the platform-specific Claude Desktop config path."""
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Windows":
        return (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )
    return Path.home() / ".config" / "claude" / "claude_desktop_config.json"


def setup_vscode(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> bool:
    print("[VS Code] Setting up .vscode/mcp.json ...")
    config = generate_vscode_config(
        project_root,
        transport=transport,
        port=port,
        launcher=launcher,
        package_spec=package_spec,
    )
    target = project_root / ".vscode" / "mcp.json"
    return safe_merge_config(target, config, "servers", SERVER_NAME)


def setup_claude_desktop(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> bool:
    target = _claude_desktop_config_path()
    print(f"[Claude Desktop] Setting up {target} ...")
    config = generate_claude_desktop_config(
        project_root,
        transport=transport,
        port=port,
        launcher=launcher,
        package_spec=package_spec,
    )
    return safe_merge_config(target, config, "mcpServers", SERVER_NAME)


def setup_claude_code(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> bool:
    print("[Claude Code] Setting up .mcp.json ...")
    config = generate_claude_code_config(
        project_root,
        transport=transport,
        port=port,
        launcher=launcher,
        package_spec=package_spec,
    )
    target = project_root / ".mcp.json"
    return safe_merge_config(target, config, "mcpServers", SERVER_NAME)


def setup_cursor(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> bool:
    print("[Cursor] Setting up .cursor/mcp.json ...")
    config = generate_cursor_config(
        project_root,
        transport=transport,
        port=port,
        launcher=launcher,
        package_spec=package_spec,
    )
    target = project_root / ".cursor" / "mcp.json"
    return safe_merge_config(target, config, "mcpServers", SERVER_NAME)


ALL_CLIENTS = {
    "vscode": ("VS Code", setup_vscode),
    "claude-desktop": ("Claude Desktop", setup_claude_desktop),
    "claude-code": ("Claude Code", setup_claude_code),
    "cursor": ("Cursor", setup_cursor),
}


def interactive_setup(
    project_root: Path,
    *,
    transport: str,
    port: int,
    launcher: str,
    package_spec: str,
) -> None:
    """Present a numbered menu and set up selected clients."""
    entries = list(ALL_CLIENTS.items())

    print("\nMCP Tuna MCP Setup")
    print("=" * 40)
    print(f"Project root: {project_root}")
    print(f"Transport:    {transport}")
    print(f"Launcher:     {launcher}")
    if launcher == "uvx":
        print(f"Package spec: {package_spec}")
    print()

    for i, (_, (label, _)) in enumerate(entries, 1):
        print(f"  {i}. {label}")
    print(f"  {len(entries) + 1}. All clients")
    print()

    choice = input("Select client(s) to configure (comma-separated numbers): ").strip()
    if not choice:
        print("No selection - exiting.")
        return

    indices: list[int] = []
    for part in choice.split(","):
        part = part.strip()
        if not part.isdigit():
            print(f"Invalid input: {part!r}")
            return
        indices.append(int(part))

    all_idx = len(entries) + 1
    if all_idx in indices:
        selected = entries
    else:
        selected = []
        for idx in indices:
            if 1 <= idx <= len(entries):
                selected.append(entries[idx - 1])
            else:
                print(f"Invalid option: {idx}")
                return

    successes = 0
    for _, (_, setup_fn) in selected:
        if setup_fn(
            project_root,
            transport=transport,
            port=port,
            launcher=launcher,
            package_spec=package_spec,
        ):
            successes += 1
    print(f"\nDone - {successes}/{len(selected)} configs written successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate MCP client configs for MCP Tuna.",
    )
    parser.add_argument("--all", action="store_true", help="Configure all clients")
    parser.add_argument("--vscode", action="store_true", help="Configure VS Code")
    parser.add_argument(
        "--claude-desktop", action="store_true", help="Configure Claude Desktop"
    )
    parser.add_argument(
        "--claude-code", action="store_true", help="Configure Claude Code"
    )
    parser.add_argument("--cursor", action="store_true", help="Configure Cursor")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="HTTP port when --transport http (default: 8002)",
    )
    parser.add_argument(
        "--launcher",
        choices=["installed", "uvx", "repo"],
        default="installed",
        help="How clients should launch MCP Tuna (default: installed).",
    )
    parser.add_argument(
        "--package-spec",
        default=DEFAULT_PACKAGE_SPEC,
        help=(
            "Package spec to use with --launcher uvx "
            f"(default: {DEFAULT_PACKAGE_SPEC})."
        ),
    )

    args = parser.parse_args()
    project_root = resolve_project_root(args.launcher)

    explicit = any(
        [
            args.all,
            args.vscode,
            args.claude_desktop,
            args.claude_code,
            args.cursor,
        ]
    )
    if not explicit:
        interactive_setup(
            project_root,
            transport=args.transport,
            port=args.port,
            launcher=args.launcher,
            package_spec=args.package_spec,
        )
        return

    targets: list[tuple[str, object]] = []
    if args.all:
        targets = [(label, fn) for label, fn in ALL_CLIENTS.values()]
    else:
        if args.vscode:
            targets.append(ALL_CLIENTS["vscode"])
        if args.claude_desktop:
            targets.append(ALL_CLIENTS["claude-desktop"])
        if args.claude_code:
            targets.append(ALL_CLIENTS["claude-code"])
        if args.cursor:
            targets.append(ALL_CLIENTS["cursor"])

    successes = 0
    for _, setup_fn in targets:
        if setup_fn(
            project_root,
            transport=args.transport,
            port=args.port,
            launcher=args.launcher,
            package_spec=args.package_spec,
        ):
            successes += 1
    print(f"\nDone - {successes}/{len(targets)} configs written successfully.")


if __name__ == "__main__":
    main()
