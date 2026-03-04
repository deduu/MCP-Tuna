"""Entry point for mcp-tuna-data (data preparation MCP server).

Usage:
    python -m scripts.run_data_server           # stdio mode
    python -m scripts.run_data_server http 8010  # HTTP mode
"""

from servers._entry import server_main
from servers.data_server import DataPrepServer


def main():
    server_main(DataPrepServer, "mcp-tuna-data", default_port=8010)


if __name__ == "__main__":
    main()
