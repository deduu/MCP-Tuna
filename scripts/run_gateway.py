"""
AgentY Gateway — Entry point for the unified MCP server.

Usage:
    # Stdio mode (for Claude Desktop / MCP clients)
    python -m AgentY.scripts.run_gateway

    # HTTP mode (for testing / web clients)
    python -m AgentY.scripts.run_gateway http 8000
"""

import sys
import os

# Ensure src/ is importable so agent_framework resolves correctly
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, "src"))

from server.mcp_server import StdioTransport, HTTPTransport
from AgentY.mcp_gateway import AgentYGateway


def main():
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
