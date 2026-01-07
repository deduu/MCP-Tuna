# ============================================================================
# FILE: scripts/run_mcp_server.py
# ============================================================================

"""
Launch MCP server for fine-tuning pipeline.

Usage:
    python scripts/run_mcp_server.py --config config.yaml
"""

import asyncio
import argparse
import yaml
import os
import sys
from dotenv import load_dotenv
from ..mcp.server import FineTuningMCPServer
from src.agent_framework.providers.openai import OpenAIProvider

load_dotenv(override=True)  # override system env if needed
model = os.getenv("OPENAI_MODEL", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")


async def main():
    parser = argparse.ArgumentParser(description="Run Fine-tuning MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        default="./AgentY/data_generator_pipeline/scripts/config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup LLM provider
    # llm = OpenAIProvider(
    #     model=config["model"],
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )
    llm = OpenAIProvider(model=model, api_key=api_key, base_url=base_url)

    # Create and run server
    server = FineTuningMCPServer(llm, config)

    print("🚀 Fine-tuning MCP Server starting...", file=sys.stderr)
    print("📝 Available tools:", file=sys.stderr)
    print("   - load_document", file=sys.stderr)
    print("   - generate_from_page", file=sys.stderr)
    print("   - generate_from_document", file=sys.stderr)
    print("   - generate_batch", file=sys.stderr)
    print("   - export_dataset", file=sys.stderr)
    print("   - list_techniques", file=sys.stderr)
    print("   - get_technique_schema", file=sys.stderr)
    print("\n✨ Server ready for connections!", file=sys.stderr)

    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
