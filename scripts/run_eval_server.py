"""Entry point for transcendence-eval (data evaluation MCP server).

Usage:
    python -m scripts.run_eval_server           # stdio mode
    python -m scripts.run_eval_server http 8011  # HTTP mode
"""

from servers._entry import server_main
from servers.eval_server import EvalServer


def main():
    server_main(EvalServer, "transcendence-eval", default_port=8011)


if __name__ == "__main__":
    main()
