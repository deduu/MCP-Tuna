"""MCP server for database tools."""

import json

from agentsoul.server import MCPServer
from .service import DatabaseService
from app.core.config import settings


class DatabaseMCPServer:
    """Exposes database operations as MCP tools."""

    def __init__(self):
        self.service = DatabaseService(settings.database.url)
        self.mcp = MCPServer("database-tools", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(
            name="db.query",
            description="Execute a read-only SQL query and return results as JSON",
        )
        async def db_query(sql: str) -> str:
            result = await svc.query(sql)
            return json.dumps(result, indent=2, default=str)

        @self.mcp.tool(
            name="db.list_tables",
            description="List all tables in the database",
        )
        async def db_list_tables() -> str:
            result = await svc.list_tables()
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="db.describe_table",
            description="Get column names, types, and constraints for a table",
        )
        async def db_describe_table(table_name: str) -> str:
            result = await svc.describe_table(table_name)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="db.insert",
            description="Insert a row into a table. Pass data as a JSON string of {column: value}",
        )
        async def db_insert(table_name: str, data: str) -> str:
            parsed = json.loads(data)
            result = await svc.insert(table_name, parsed)
            return json.dumps(result, indent=2, default=str)

    def run(self, transport=None):
        self.mcp.run(transport)
