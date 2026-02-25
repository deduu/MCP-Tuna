# app/core/agent_factory.py
from typing import Optional, List, Dict
from agentsoul.core.agent import AgentSoul
from app.generation.registry import get_registry
from app.core.config import settings


async def create_agent(
    model_name: str,
    mcp_servers: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_turns: int = 10,
    system_prompt: Optional[str] = None,
) -> AgentSoul:
    """
    Create a AgentSoul instance, optionally wired to MCP tool servers.

    If mcp_servers is provided, those are used directly.
    Otherwise, auto-connects to servers defined in settings.mcp when auto_connect=True.
    """
    registry = get_registry()
    llm_provider = await registry.get(model_name, api_key=api_key, base_url=base_url)

    tools = _build_tool_configs(mcp_servers)

    if tools:
        agent = await AgentSoul.create(
            llm_provider=llm_provider,
            tools=tools,
            system_prompt=system_prompt,
            max_turns=max_turns,
        )
    else:
        agent = AgentSoul(
            llm_provider=llm_provider,
            system_prompt=system_prompt,
            max_turns=max_turns,
        )

    return agent


def _build_tool_configs(
    mcp_servers: Optional[List[Dict]] = None,
) -> Optional[List[Dict]]:
    """Convert MCP server configs into tool config dicts for AgentSoul.create()."""
    tools = []

    # Use explicit MCP servers if provided
    if mcp_servers:
        for server in mcp_servers:
            tools.append({
                "type": "mcp",
                "server_label": server["server_label"],
                "server_url": server["server_url"],
                "server_description": server.get("server_description", ""),
                "require_approval": server.get("require_approval", "never"),
            })
        return tools

    # Auto-connect to configured servers
    if settings.mcp.auto_connect:
        for server_cfg in settings.mcp.servers:
            tools.append({
                "type": "mcp",
                "server_label": server_cfg.server_label,
                "server_url": server_cfg.server_url,
                "server_description": server_cfg.server_description,
                "require_approval": server_cfg.require_approval,
            })

    return tools if tools else None
