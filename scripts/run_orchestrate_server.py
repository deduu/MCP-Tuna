"""Entry point for mcp-tuna-orchestrate (orchestration MCP server).

Usage:
    python -m scripts.run_orchestrate_server           # stdio mode
    python -m scripts.run_orchestrate_server http 8015  # HTTP mode
"""

from servers._entry import server_main
from servers.orchestrate_server import OrchestrateServer


def main():
    server_main(OrchestrateServer, "mcp-tuna-orchestrate", default_port=8015)


if __name__ == "__main__":
    main()
