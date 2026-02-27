"""Entry point for transcendence-train (training MCP server).

Usage:
    python -m scripts.run_train_server           # stdio mode
    python -m scripts.run_train_server http 8013  # HTTP mode
"""

from servers._entry import server_main
from servers.train_server import TrainServer


def main():
    server_main(TrainServer, "transcendence-train", default_port=8013)


if __name__ == "__main__":
    main()
