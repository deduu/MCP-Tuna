"""Entry point for mcp-tuna-host (model hosting MCP server).

Usage:
    python -m scripts.run_host_server           # stdio mode
    python -m scripts.run_host_server http 8014  # HTTP mode
"""

from servers._entry import server_main
from servers.host_server import HostServer


def main():
    server_main(HostServer, "mcp-tuna-host", default_port=8014)


if __name__ == "__main__":
    main()
