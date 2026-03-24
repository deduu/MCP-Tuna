"""Shared entry point logic for standalone MCP servers."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

from shared.version import get_package_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _prefer_workspace_sources() -> None:
    project_root_str = str(PROJECT_ROOT)
    src_root_str = str(SRC_ROOT)
    for path in (src_root_str, project_root_str):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, project_root_str)
    sys.path.insert(0, src_root_str)


def server_main(server_cls, name: str, default_port: int = 8000, **init_kwargs):
    """Standard entry point for all MCP Tuna standalone MCP servers.

    Usage:
        mcp-tuna-<name>              # stdio mode
        mcp-tuna-<name> http         # HTTP mode, default port
        mcp-tuna-<name> http --port 9000
        mcp-tuna-<name> --version
    """
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    _prefer_workspace_sources()

    from agentsoul.server import HTTPTransport, StdioTransport
    from agentsoul.utils.logger import configure_logging
    from dotenv import load_dotenv

    load_dotenv(override=False)
    configure_logging(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog=name,
        description=f"{name} - MCP Tuna standalone MCP server.",
    )
    package_version = get_package_version()
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {package_version}"
    )
    sub = parser.add_subparsers(dest="transport")

    http_parser = sub.add_parser("http", help="Start in HTTP mode")
    http_parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to listen on (default: {default_port})",
    )
    http_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    try:
        server = server_cls(**init_kwargs)
    except Exception as exc:
        print(f"ERROR: failed to initialize {name}: {exc}", flush=True)
        raise SystemExit(1) from exc

    if args.transport == "http":
        _check_port(args.host, args.port)
        transport = HTTPTransport(host=args.host, port=args.port)
        print(f"Starting {name} on http://{args.host}:{args.port}", flush=True)
    else:
        transport = StdioTransport()
        print(f"Starting {name} (stdio mode)", file=sys.stderr)

    try:
        server.run(transport)
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        return
    except Exception as exc:
        print(f"ERROR: {name} crashed: {exc}", flush=True)
        raise SystemExit(1) from exc


def _check_port(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0:
            print(f"ERROR: port {port} is already in use.", flush=True)
            raise SystemExit(1)
