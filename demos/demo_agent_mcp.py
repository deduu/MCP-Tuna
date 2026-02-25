"""
Demo: AgentSoul + AgentY MCP Gateway
========================================

Shows an AI agent autonomously using AgentY pipeline tools via MCP.

Prerequisites:
    - OPENAI_API_KEY set in environment (or .env file)
    - AgentY Gateway already running:
        python -m AgentY.scripts.run_gateway http 8002

Usage:
    # Connect to gateway on default port 8002
    python demos/demo_agent_mcp.py

    # Custom prompt
    python demos/demo_agent_mcp.py "Generate SFT data from ./data/sample.pdf"

    # Custom gateway port
    python demos/demo_agent_mcp.py --port 9000 "List all tools"

    # Connect to multiple MCP servers (gateway + web tools)
    python demos/demo_agent_mcp.py --web "Search the web for LLaMA fine-tuning"
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import auditi
    auditi.init(
        api_key=os.getenv("AUDITI_API_KEY"),
        base_url="http://localhost:8000",
    )
except ImportError:
    pass


DEFAULT_GATEWAY_PORT = int(os.getenv("AGENTY_GATEWAY_PORT", "8002"))
DEFAULT_PROMPT = "List the available fine-tuning techniques, then explain what SFT is."


def parse_args():
    """Parse CLI arguments."""
    port = DEFAULT_GATEWAY_PORT
    include_web = False
    prompt = DEFAULT_PROMPT

    args = sys.argv[1:]
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--web":
            include_web = True
            i += 1
        else:
            positional.append(args[i])
            i += 1

    if positional:
        prompt = " ".join(positional)

    return port, include_web, prompt


async def check_server(url: str, label: str) -> bool:
    """Check if an MCP server is reachable."""
    import httpx

    health_url = url.replace("/mcp", "/health")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(health_url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                tools = data.get("tools", "?")
                print(f"  {label}: {health_url} ({tools} tools)")
                return True
    except (httpx.ConnectError, httpx.ReadTimeout):
        pass
    print(f"  {label}: NOT REACHABLE at {health_url}")
    return False


async def run_agent(prompt: str, mcp_tools: list):
    """Create a AgentSoul wired to MCP servers and run a prompt."""
    from agentsoul.providers.openai import OpenAIProvider
    from agentsoul.core.agent import AgentSoul

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Export it or add to .env file.")
        return

    llm = OpenAIProvider(
        model_id="gpt-4o",
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    print(f"\n  Creating agent with {len(mcp_tools)} MCP server(s)...")
    agent = await AgentSoul.create(
        llm_provider=llm,
        tools=mcp_tools,
        system_prompt=(
            "You are a data engineering agent with access to pipeline tools via MCP. "
            "Use the available tools to help users with data generation, cleaning, "
            "evaluation, fine-tuning, and model hosting tasks. "
            "Always use tools when they can help answer the question."
        ),
        max_turns=10,
    )

    print(f"\n  Prompt: {prompt}\n")
    print("=" * 60)
    print()

    async for event in agent.run(prompt, enable_streaming=True):
        event_type = event.get("type", "")

        if event_type == "token":
            print(event["content"], end="", flush=True)
        elif event_type == "tool_exec_start":
            print(f"\n  [TOOL] Calling: {event.get('tool', '?')}...")
        elif event_type == "tool_exec_end":
            print(f"  [TOOL] Done: {event.get('tool', '?')}")
        elif event_type == "complete":
            print("\n")
            print("=" * 60)
            print("  Agent finished.")
            break


async def main():
    port, include_web, prompt = parse_args()
    gateway_url = f"http://localhost:{port}/mcp"

    # Build MCP tool configs
    mcp_tools = [{
        "type": "mcp",
        "server_label": "agenty-gateway",
        "server_url": gateway_url,
        "server_description": "AgentY data pipeline tools",
    }]

    if include_web:
        web_port = int(os.getenv("WEB_MCP_PORT", "8005"))
        mcp_tools.append({
            "type": "mcp",
            "server_label": "web-tools",
            "server_url": f"http://localhost:{web_port}/mcp",
            "server_description": "Web fetch and search tools",
        })

    # Check connectivity
    print("Checking MCP servers...")
    all_ok = True
    for tool in mcp_tools:
        ok = await check_server(tool["server_url"], tool["server_label"])
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nSome servers are not reachable. Start them first:")
        print(f"  python -m AgentY.scripts.run_gateway http {port}")
        if include_web:
            print(f"  python -m app.services.run_servers web")
        print()
        return

    await run_agent(prompt, mcp_tools)


if __name__ == "__main__":
    asyncio.run(main())
