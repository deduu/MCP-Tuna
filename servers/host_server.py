"""Hosting MCP server wrapper.

Re-exports the existing HostingMCPServer with host.health added.
"""

from __future__ import annotations

import json
from typing import Optional

from agentsoul.server import MCPServer
from hosting_pipeline.services.hosting_service import HostingService
from shared.config import HostingConfig


class HostServer:
    """Model deployment tools with health checking."""

    def __init__(self):
        self.service = HostingService()
        self.mcp = MCPServer("mcp-tuna-host", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(name="host.deploy_mcp",
                       description="Deploy a fine-tuned model as an MCP tool server")
        async def deploy_mcp(
            model_path: str, adapter_path: Optional[str] = None, port: int = 8001,
        ) -> str:
            config = HostingConfig(model_path=model_path, adapter_path=adapter_path, port=port)
            return json.dumps(await svc.deploy_as_mcp(config), indent=2)

        @self.mcp.tool(name="host.deploy_vlm_mcp",
                       description="Deploy a vision-language model as an MCP tool server")
        async def deploy_vlm_mcp(
            model_path: str, adapter_path: Optional[str] = None, port: int = 8001,
        ) -> str:
            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                port=port,
                modality="vision-language",
            )
            return json.dumps(await svc.deploy_vlm_as_mcp(config), indent=2)

        @self.mcp.tool(name="host.deploy_api",
                       description="Deploy a fine-tuned model as a REST API")
        async def deploy_api(
            model_path: str, adapter_path: Optional[str] = None, port: int = 8001,
        ) -> str:
            config = HostingConfig(model_path=model_path, adapter_path=adapter_path, port=port)
            return json.dumps(await svc.deploy_as_api(config), indent=2)

        @self.mcp.tool(name="host.deploy_vlm_api",
                       description="Deploy a vision-language model as a REST API")
        async def deploy_vlm_api(
            model_path: str, adapter_path: Optional[str] = None, port: int = 8001,
        ) -> str:
            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                port=port,
                modality="vision-language",
            )
            return json.dumps(await svc.deploy_vlm_as_api(config), indent=2)

        @self.mcp.tool(name="host.list_deployments",
                       description="List running model deployments")
        async def list_deployments() -> str:
            return json.dumps(await svc.list_deployments(), indent=2)

        @self.mcp.tool(name="host.stop", description="Stop a running deployment")
        async def stop_deployment(deployment_id: str) -> str:
            return json.dumps(await svc.stop_deployment(deployment_id), indent=2)

        @self.mcp.tool(name="host.health",
                       description="Health check on a running deployment")
        async def health(deployment_id: str) -> str:
            return json.dumps(await svc.health_check(deployment_id), indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
