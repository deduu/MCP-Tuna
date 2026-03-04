"""Hosting service — deploy fine-tuned models as MCP tool servers or REST APIs."""

import asyncio
import uuid
from typing import Any, Dict

from shared.config import HostingConfig


class HostingService:
    """Manages model deployments as MCP servers or FastAPI endpoints."""

    def __init__(self):
        self._deployments: Dict[str, Dict[str, Any]] = {}

    async def deploy_as_mcp(self, config: HostingConfig) -> Dict[str, Any]:
        """Deploy a fine-tuned model as an MCP tool server."""
        try:
            from agentsoul.server import MCPServer, HTTPTransport, StdioTransport
            from agentsoul.providers.hf import HuggingFaceProvider

            provider = HuggingFaceProvider(model_path=config.model_path)
            if config.adapter_path:
                provider.load_adapter(config.adapter_path)

            mcp = MCPServer(f"hosted-model-{config.port}", "1.0.0")

            @mcp.tool(name="generate",
                      description=f"Generate text with fine-tuned model at {config.model_path}")
            async def generate(prompt: str, max_new_tokens: int = 512) -> str:
                resp = await provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_new_tokens=max_new_tokens,
                )
                return resp.content or ""

            deployment_id = str(uuid.uuid4())[:8]

            if config.transport == "http":
                transport = HTTPTransport(host=config.host, port=config.port)
            else:
                transport = StdioTransport()

            # Start in background
            task = asyncio.create_task(mcp.run_async(transport))

            self._deployments[deployment_id] = {
                "id": deployment_id,
                "model_path": config.model_path,
                "adapter_path": config.adapter_path,
                "transport": config.transport,
                "host": config.host,
                "port": config.port,
                "task": task,
                "mcp": mcp,
            }

            return {
                "success": True,
                "deployment_id": deployment_id,
                "model_path": config.model_path,
                "transport": config.transport,
                "endpoint": f"http://{config.host}:{config.port}" if config.transport == "http" else "stdio",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def deploy_as_api(self, config: HostingConfig) -> Dict[str, Any]:
        """Deploy as a FastAPI endpoint with /generate POST route."""
        try:
            from fastapi import FastAPI
            import uvicorn
            from agentsoul.providers.hf import HuggingFaceProvider

            provider = HuggingFaceProvider(model_path=config.model_path)
            if config.adapter_path:
                provider.load_adapter(config.adapter_path)

            app = FastAPI(title=f"Transcendence Model: {config.model_path}")

            @app.post("/generate")
            async def generate(prompt: str, max_new_tokens: int = 512):
                resp = await provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_new_tokens=max_new_tokens,
                )
                return {"response": resp.content or ""}

            @app.get("/health")
            async def health():
                return {"status": "ok", "model": config.model_path}

            deployment_id = str(uuid.uuid4())[:8]

            loop = asyncio.get_running_loop()
            server_config = uvicorn.Config(app, host=config.host, port=config.port, log_level="info")
            server = uvicorn.Server(server_config)
            task = loop.create_task(server.serve())

            self._deployments[deployment_id] = {
                "id": deployment_id,
                "model_path": config.model_path,
                "adapter_path": config.adapter_path,
                "transport": "http",
                "host": config.host,
                "port": config.port,
                "task": task,
                "server": server,
            }

            return {
                "success": True,
                "deployment_id": deployment_id,
                "model_path": config.model_path,
                "endpoint": f"http://{config.host}:{config.port}",
                "routes": ["/generate", "/health"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_deployments(self) -> Dict[str, Any]:
        """List currently running model deployments."""
        deployments = []
        for dep_id, dep in self._deployments.items():
            deployments.append({
                "id": dep_id,
                "model_path": dep["model_path"],
                "adapter_path": dep.get("adapter_path"),
                "transport": dep["transport"],
                "endpoint": f"http://{dep['host']}:{dep['port']}" if dep["transport"] == "http" else "stdio",
            })
        return {"success": True, "deployments": deployments, "count": len(deployments)}

    async def stop_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Stop a running deployment."""
        if deployment_id not in self._deployments:
            return {"success": False, "error": f"Deployment {deployment_id} not found"}

        dep = self._deployments.pop(deployment_id)
        task = dep.get("task")
        if task and not task.done():
            task.cancel()

        server = dep.get("server")
        if server:
            server.should_exit = True

        return {"success": True, "deployment_id": deployment_id, "status": "stopped"}

    async def health_check(self, deployment_id: str) -> Dict[str, Any]:
        """Check health of a running deployment."""
        if deployment_id not in self._deployments:
            return {"success": False, "error": f"Deployment {deployment_id} not found"}

        dep = self._deployments[deployment_id]
        task = dep.get("task")
        is_alive = task is not None and not task.done()

        result: Dict[str, Any] = {
            "success": True,
            "deployment_id": deployment_id,
            "status": "running" if is_alive else "stopped",
            "model_path": dep.get("model_path"),
            "endpoint": (
                f"http://{dep['host']}:{dep['port']}"
                if dep.get("transport") == "http" or dep.get("server")
                else "stdio"
            ),
        }

        # For API deployments, try hitting the /health endpoint
        if is_alive and dep.get("server"):
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"http://{dep['host']}:{dep['port']}/health")
                    result["health_response"] = resp.json()
            except Exception:
                result["health_response"] = None

        return result
