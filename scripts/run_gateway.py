"""
AgentY Gateway — Entry point for the unified MCP server.

Usage:
    # Stdio mode (for Claude Desktop / MCP clients)
    python -m AgentY.scripts.run_gateway

    # HTTP mode (for testing / web clients)
    python -m AgentY.scripts.run_gateway http 8000
"""

import sys
import logging

# Ensure stderr uses UTF-8 on Windows to prevent UnicodeEncodeError in log handlers
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agentsoul.server import StdioTransport, HTTPTransport
from agentsoul.utils.logger import configure_logging
from mcp_gateway import AgentYGateway


def main():
    configure_logging(level=logging.DEBUG)
    gateway = AgentYGateway()

    if len(sys.argv) > 1 and sys.argv[1] == "http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        transport = HTTPTransport(host="0.0.0.0", port=port)
        print(f"Starting AgentY Gateway on http://0.0.0.0:{port}", file=sys.stderr)
        print(f"Health check: http://localhost:{port}/health", file=sys.stderr)
    else:
        transport = StdioTransport()
        print("Starting AgentY Gateway (stdio mode)", file=sys.stderr)

    gateway.run(transport)


if __name__ == "__main__":
    main()
