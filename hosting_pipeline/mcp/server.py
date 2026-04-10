"""MCP server for model hosting tools."""

import json
from typing import Any, Dict, Optional

from agentsoul.server import MCPServer
from ..services.hosting_service import HostingService
from shared.config import HostingConfig, ThinkingMode


def _parse_json_option(raw: Optional[str], field_name: str) -> Any:
    if raw is None or not str(raw).strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {field_name}: {exc}") from exc


def _build_inference_placement(
    *,
    device_map: Optional[str],
    device_map_json: Optional[str],
    max_memory_json: Optional[str],
    offload_folder: Optional[str],
) -> Optional[Dict[str, Any]]:
    placement = {
        key: value
        for key, value in {
            "device_map": _parse_json_option(device_map_json, "device_map_json")
            if device_map_json
            else device_map,
            "max_memory": _parse_json_option(max_memory_json, "max_memory_json"),
            "offload_folder": offload_folder,
        }.items()
        if value is not None
    }
    return placement or None


class HostingMCPServer:
    """Exposes model hosting operations as MCP tools."""

    def __init__(self):
        self.service = HostingService()
        self.mcp = MCPServer("mcp-tuna-hosting", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(name="host.deploy_mcp",
                       description="Deploy a fine-tuned model as an MCP tool server")
        async def deploy_as_mcp(
            model_path: str,
            adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
            host: str = "0.0.0.0",
            thinking_mode: ThinkingMode = "default",
            device_map: Optional[str] = None,
            device_map_json: Optional[str] = None,
            max_memory_json: Optional[str] = None,
            offload_folder: Optional[str] = None,
        ) -> str:
            try:
                placement = _build_inference_placement(
                    device_map=device_map,
                    device_map_json=device_map_json,
                    max_memory_json=max_memory_json,
                    offload_folder=offload_folder,
                )
            except ValueError as exc:
                return json.dumps({"success": False, "error": str(exc)}, indent=2)

            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                name=name,
                host=host,
                port=port,
                transport="http",
                thinking_mode=thinking_mode,
                inference_placement=placement or None,
            )
            result = await svc.deploy_as_mcp(config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="host.deploy_vlm_mcp",
                       description="Deploy a vision-language model as an MCP tool server")
        async def deploy_vlm_as_mcp(
            model_path: str,
            adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
            host: str = "0.0.0.0",
            device_map: Optional[str] = None,
            device_map_json: Optional[str] = None,
            max_memory_json: Optional[str] = None,
            offload_folder: Optional[str] = None,
        ) -> str:
            try:
                placement = _build_inference_placement(
                    device_map=device_map,
                    device_map_json=device_map_json,
                    max_memory_json=max_memory_json,
                    offload_folder=offload_folder,
                )
            except ValueError as exc:
                return json.dumps({"success": False, "error": str(exc)}, indent=2)

            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                name=name,
                host=host,
                port=port,
                transport="http",
                modality="vision-language",
                inference_placement=placement or None,
            )
            result = await svc.deploy_vlm_as_mcp(config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="host.deploy_api",
                       description="Deploy a fine-tuned model as a REST API with /generate endpoint")
        async def deploy_as_api(
            model_path: str,
            adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
            host: str = "0.0.0.0",
            thinking_mode: ThinkingMode = "default",
            device_map: Optional[str] = None,
            device_map_json: Optional[str] = None,
            max_memory_json: Optional[str] = None,
            offload_folder: Optional[str] = None,
        ) -> str:
            try:
                placement = _build_inference_placement(
                    device_map=device_map,
                    device_map_json=device_map_json,
                    max_memory_json=max_memory_json,
                    offload_folder=offload_folder,
                )
            except ValueError as exc:
                return json.dumps({"success": False, "error": str(exc)}, indent=2)

            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                name=name,
                host=host,
                port=port,
                transport="http",
                thinking_mode=thinking_mode,
                inference_placement=placement or None,
            )
            result = await svc.deploy_as_api(config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="host.deploy_vlm_api",
                       description="Deploy a vision-language model as a REST API with /generate_vlm endpoint")
        async def deploy_vlm_as_api(
            model_path: str,
            adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
            host: str = "0.0.0.0",
            device_map: Optional[str] = None,
            device_map_json: Optional[str] = None,
            max_memory_json: Optional[str] = None,
            offload_folder: Optional[str] = None,
        ) -> str:
            try:
                placement = _build_inference_placement(
                    device_map=device_map,
                    device_map_json=device_map_json,
                    max_memory_json=max_memory_json,
                    offload_folder=offload_folder,
                )
            except ValueError as exc:
                return json.dumps({"success": False, "error": str(exc)}, indent=2)

            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                name=name,
                host=host,
                port=port,
                transport="http",
                modality="vision-language",
                inference_placement=placement or None,
            )
            result = await svc.deploy_vlm_as_api(config)
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

        @self.mcp.tool(name="host.delete_deployment",
                       description="Delete a deployment record and stop it first if it is still running")
        async def delete_deployment(deployment_id: str) -> str:
            result = await svc.delete_deployment(deployment_id)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
