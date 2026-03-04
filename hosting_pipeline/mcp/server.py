"""MCP server for model hosting tools."""

import json
from typing import Optional

from agentsoul.server import MCPServer
from ..services.hosting_service import HostingService
from shared.config import HostingConfig


class HostingMCPServer:
    """Exposes model hosting operations as MCP tools."""

    def __init__(self):
        self.service = HostingService()
        self.mcp = MCPServer("transcendence-hosting", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(name="host.deploy_mcp",
                       description="Deploy a fine-tuned model as an MCP tool server")
        async def deploy_as_mcp(
            model_path: str,
            adapter_path: Optional[str] = None,
            port: int = 8001,
            host: str = "0.0.0.0",
        ) -> str:
            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                host=host,
                port=port,
                transport="http",
            )
            result = await svc.deploy_as_mcp(config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="host.deploy_api",
                       description="Deploy a fine-tuned model as a REST API with /generate endpoint")
        async def deploy_as_api(
            model_path: str,
            adapter_path: Optional[str] = None,
            port: int = 8001,
            host: str = "0.0.0.0",
        ) -> str:
            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                host=host,
                port=port,
                transport="http",
            )
            result = await svc.deploy_as_api(config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="host.list_deployments",
                       description="List all currently running model deployments")
        async def list_deployments() -> str:
            result = await svc.list_deployments()
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="host.stop",
                       description="Stop a running model deployment")
        async def stop_deployment(deployment_id: str) -> str:
            result = await svc.stop_deployment(deployment_id)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
