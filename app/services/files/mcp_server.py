"""MCP server for file management tools."""

import json

from agentsoul.server import MCPServer
from .service import FileService
from app.core.config import settings


class FileMCPServer:
    """Exposes file management operations as MCP tools."""

    def __init__(self):
        self.service = FileService(str(settings.files.upload_root))
        self.mcp = MCPServer("file-tools", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(
            name="file.read",
            description="Read a file's contents. Returns text for text files, base64 for binary.",
        )
        async def file_read(path: str) -> str:
            result = await svc.read(path)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="file.write",
            description="Write text content to a file. Creates parent directories if needed.",
        )
        async def file_write(path: str, content: str) -> str:
            result = await svc.write(path, content)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="file.list",
            description="List files and directories at a path.",
        )
        async def file_list(path: str = ".") -> str:
            result = await svc.list_dir(path)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="file.upload",
            description="Save a base64-encoded file to the uploads directory.",
        )
        async def file_upload(filename: str, content_base64: str) -> str:
            result = await svc.upload(filename, content_base64)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
