"""Entry point for transcendence-model-eval (model evaluation + judge MCP server).

Usage:
    python -m scripts.run_model_eval_server           # stdio mode
    python -m scripts.run_model_eval_server http 8012  # HTTP mode
"""

from servers._entry import server_main
from servers.model_eval_server import ModelEvalServer


def main():
    server_main(ModelEvalServer, "transcendence-model-eval", default_port=8012)


if __name__ == "__main__":
    main()
