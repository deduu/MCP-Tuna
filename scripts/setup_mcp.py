"""Auto-generate MCP client configuration files for MCP Tuna.

Supports VS Code, Claude Desktop, Claude Code, and Cursor.

Usage:
    python scripts/setup_mcp.py              # Interactive menu
    python scripts/setup_mcp.py --all        # All clients
    python scripts/setup_mcp.py --vscode     # Just VS Code
    python scripts/setup_mcp.py --claude-desktop
    python scripts/setup_mcp.py --claude-code
    python scripts/setup_mcp.py --cursor
    python scripts/setup_mcp.py --transport http  # HTTP instead of stdio
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


def detect_project_root() -> Path:
    """Walk up from this script's directory to find the dir containing mcp_gateway.py."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "mcp_gateway.py").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    print("Error: Could not find project root (directory containing mcp_gateway.py).", file=sys.stderr)
    sys.exit(1)


def build_stdio_entry(project_root: Path, *, needs_abs_path: bool = False) -> dict:
    """Build a stdio-mode server entry dict.

    Args:
        project_root: Absolute path to the MCP Tuna project root.
        needs_abs_path: If True, uses ``uv --directory <abs>`` for clients
            that don't support CWD (e.g. Claude Desktop).
    """
    if needs_abs_path:
        return {
            "command": "uv",
            "args": [
                "--directory",
                str(project_root),
                "run",
                "python",
                "scripts/run_gateway.py",
            ],
        }
    return {
        "command": "uv",
        "args": ["run", "python", "scripts/run_gateway.py"],
    }


def build_http_entry(port: int = 8002) -> dict:
    """Build an HTTP-mode server entry dict."""
    return {
        "type": "http",
        "url": f"http://localhost:{port}/mcp",
    }


SERVER_NAME = "mcp-tuna-gateway"


# ---------------------------------------------------------------------------
# Per-client config generators
# ---------------------------------------------------------------------------

def generate_vscode_config(project_root: Path, *, transport: str, port: int) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(project_root)
    return {"servers": {SERVER_NAME: entry}}


def generate_claude_desktop_config(project_root: Path, *, transport: str, port: int) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(project_root, needs_abs_path=True)
    return {"mcpServers": {SERVER_NAME: entry}}


def generate_claude_code_config(project_root: Path, *, transport: str, port: int) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(project_root)
    return {"mcpServers": {SERVER_NAME: entry}}


def generate_cursor_config(project_root: Path, *, transport: str, port: int) -> dict:
    if transport == "http":
        entry = build_http_entry(port)
    else:
        entry = build_stdio_entry(project_root)
    return {"mcpServers": {SERVER_NAME: entry}}


# ---------------------------------------------------------------------------
# Safe merge logic
# ---------------------------------------------------------------------------

def safe_merge_config(
    path: Path,
    new_config: dict,
    server_key: str,
    server_name: str,
) -> bool:
    """Read existing config, back it up, merge only our server entry, and write.

    Args:
        path: Target config file path.
        new_config: Full config dict we'd write for a fresh file.
        server_key: Top-level key holding servers (``"servers"`` or ``"mcpServers"``).
        server_name: Our server name inside that key.

    Returns:
        True on success, False on failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            # Empty file — treat as fresh
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

        # Backup before modifying
        backup = path.with_suffix(path.suffix + ".backup")
        shutil.copy2(path, backup)
        print(f"  Backed up existing config to {backup}")

        # Merge: ensure top-level key exists, then insert/overwrite our entry
        if server_key not in existing:
            existing[server_key] = {}
        existing[server_key][server_name] = new_config[server_key][server_name]
        merged = existing
    else:
        merged = new_config

    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {path}")
    return True


# ---------------------------------------------------------------------------
# Per-client setup orchestrators
# ---------------------------------------------------------------------------

def _claude_desktop_config_path() -> Path:
    """Return the platform-specific Claude Desktop config path."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = Path.home() / "AppData" / "Roaming" / "Claude"
        return appdata / "claude_desktop_config.json"
    # Linux / fallback
    return Path.home() / ".config" / "claude" / "claude_desktop_config.json"


def setup_vscode(project_root: Path, *, transport: str, port: int) -> bool:
    print("[VS Code] Setting up .vscode/mcp.json ...")
    config = generate_vscode_config(project_root, transport=transport, port=port)
    target = project_root / ".vscode" / "mcp.json"
    return safe_merge_config(target, config, "servers", SERVER_NAME)


def setup_claude_desktop(project_root: Path, *, transport: str, port: int) -> bool:
    target = _claude_desktop_config_path()
    print(f"[Claude Desktop] Setting up {target} ...")
    config = generate_claude_desktop_config(project_root, transport=transport, port=port)
    return safe_merge_config(target, config, "mcpServers", SERVER_NAME)


def setup_claude_code(project_root: Path, *, transport: str, port: int) -> bool:
    print("[Claude Code] Setting up .mcp.json ...")
    config = generate_claude_code_config(project_root, transport=transport, port=port)
    target = project_root / ".mcp.json"
    return safe_merge_config(target, config, "mcpServers", SERVER_NAME)


def setup_cursor(project_root: Path, *, transport: str, port: int) -> bool:
    print("[Cursor] Setting up .cursor/mcp.json ...")
    config = generate_cursor_config(project_root, transport=transport, port=port)
    target = project_root / ".cursor" / "mcp.json"
    return safe_merge_config(target, config, "mcpServers", SERVER_NAME)


ALL_CLIENTS = {
    "vscode": ("VS Code", setup_vscode),
    "claude-desktop": ("Claude Desktop", setup_claude_desktop),
    "claude-code": ("Claude Code", setup_claude_code),
    "cursor": ("Cursor", setup_cursor),
}


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_setup(project_root: Path, *, transport: str, port: int) -> None:
    """Present a numbered menu and set up selected clients."""
    entries = list(ALL_CLIENTS.items())

    print("\nMCP Tuna MCP Setup")
    print("=" * 40)
    print(f"Project root: {project_root}")
    print(f"Transport:    {transport}\n")
    for i, (_, (label, _)) in enumerate(entries, 1):
        print(f"  {i}. {label}")
    print(f"  {len(entries) + 1}. All clients")
    print()

    choice = input("Select client(s) to configure (comma-separated numbers): ").strip()
    if not choice:
        print("No selection — exiting.")
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
        if setup_fn(project_root, transport=transport, port=port):
            successes += 1
    print(f"\nDone — {successes}/{len(selected)} configs written successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate MCP client configs for MCP Tuna.",
    )
    parser.add_argument("--all", action="store_true", help="Configure all clients")
    parser.add_argument("--vscode", action="store_true", help="Configure VS Code")
    parser.add_argument("--claude-desktop", action="store_true", help="Configure Claude Desktop")
    parser.add_argument("--claude-code", action="store_true", help="Configure Claude Code")
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

    args = parser.parse_args()
    project_root = detect_project_root()

    # Determine which clients to set up
    explicit = any([args.all, args.vscode, args.claude_desktop, args.claude_code, args.cursor])
    if not explicit:
        interactive_setup(project_root, transport=args.transport, port=args.port)
        return

    targets: list[tuple[str, type]] = []
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
        if setup_fn(project_root, transport=args.transport, port=args.port):
            successes += 1
    print(f"\nDone — {successes}/{len(targets)} configs written successfully.")


if __name__ == "__main__":
    main()
