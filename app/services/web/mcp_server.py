"""MCP server for web tools."""

import json

from agentsoul.server import MCPServer
from .service import WebService


class WebMCPServer:
    """Exposes web fetch and search as MCP tools."""

    def __init__(self):
        self.service = WebService()
        self.mcp = MCPServer("web-tools", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(
            name="web.fetch",
            description="Fetch a URL and return its content as text. HTML is automatically converted to readable text.",
        )
        async def web_fetch(url: str, max_length: int = 50000) -> str:
            result = await svc.fetch(url, max_length)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="web.search",
            description="Search the web and return results with titles, URLs, and snippets.",
        )
        async def web_search(query: str, max_results: int = 5) -> str:
            result = await svc.search(query, max_results)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
