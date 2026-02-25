from typing import Optional, List, Dict, Any


class ToolSelector:
    """Manages tool selection logic."""

    @staticmethod
    def select_tools(route: str, default_tools: Optional[List[str]]) -> Optional[List[str]]:
        """Select appropriate tools based on route."""
        if route == "follow_up_and_chat_history":
            return None
        return default_tools

    @staticmethod
    def select_mcp_servers(route: str) -> Optional[List[Dict[str, Any]]]:
        """Select MCP servers based on route.

        Returns None for chat-only routes (no tools needed).
        Returns configured MCP servers for routes that benefit from tools.
        """
        if route == "follow_up_and_chat_history":
            return None

        from app.core.config import settings

        if not settings.mcp.auto_connect:
            return None

        return [
            {
                "server_label": s.server_label,
                "server_url": s.server_url,
                "server_description": s.server_description,
                "require_approval": s.require_approval,
            }
            for s in settings.mcp.servers
        ]
