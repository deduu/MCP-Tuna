"""
MCP Tuna Gateway — Entry point for the unified MCP server.

Usage:
    mcp-tuna-gateway              # stdio mode (Claude Desktop)
    mcp-tuna-gateway http         # HTTP mode, default port 8000
    mcp-tuna-gateway http --port 9000
    mcp-tuna-gateway --version
    mcp-tuna-gateway --help
"""

import argparse
import sys
import logging
import socket

# Ensure stderr uses UTF-8 on Windows to prevent UnicodeEncodeError in log handlers
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agentsoul.server import StdioTransport, HTTPTransport
from agentsoul.utils.logger import configure_logging
from mcp_gateway import TunaGateway

__version__ = "0.1.0"


def _check_port(host: str, port: int) -> None:
    """Raise if the port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0:
            print(
                f"ERROR: port {port} is already in use. "
                f"Kill the existing process or choose another port.",
                flush=True,
            )
            raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-tuna-gateway",
        description="MCP Tuna — Unified MCP gateway for LLM fine-tuning tools.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="transport")

    http_parser = sub.add_parser("http", help="Start in HTTP mode")
    http_parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    http_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    return parser


def main():
    args = _build_parser().parse_args()
    configure_logging(level=logging.DEBUG)

    try:
        gateway = TunaGateway()
    except Exception as exc:
        print(f"ERROR: failed to initialize gateway: {exc}", flush=True)
        raise SystemExit(1) from exc

    if args.transport == "http":
        _check_port(args.host, args.port)
        transport = HTTPTransport(host=args.host, port=args.port)
        print(f"Starting MCP Tuna Gateway on http://{args.host}:{args.port}", flush=True)
        print(f"Health check: http://localhost:{args.port}/health", flush=True)
    else:
        transport = StdioTransport()
        print("Starting MCP Tuna Gateway (stdio mode)", file=sys.stderr)

    try:
        gateway.run(transport)
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        return
    except Exception as exc:
        print(f"ERROR: gateway crashed: {exc}", flush=True)
        raise SystemExit(1) from exc

    # If we reach here without exception, the server exited on its own —
    # this is the "silent failure" path (e.g. port conflict inside uvicorn).
    print(
        "ERROR: server exited unexpectedly. "
        "Check if the port is already in use.",
        flush=True,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
