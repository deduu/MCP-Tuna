"""Hosting service — deploy fine-tuned models as MCP tool servers or REST APIs."""

import asyncio
import logging
import socket
import threading
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from shared.config import HostingConfig
from shared.gpu_lock import GPULock

log = logging.getLogger(__name__)


def _quantization_to_provider_arg(quantization: str | None) -> Dict[str, Any]:
    """Map HostingConfig.quantization to HuggingFaceProvider kwargs."""
    if quantization in ("4bit", "8bit", "bitsandbytes"):
        return {"quantization": "bitsandbytes"}
    return {}


def _resolve_hf_cache_path(model_path: str) -> str:
    """If model_path points to an HF cache wrapper dir (blobs/refs/snapshots),
    resolve to the latest snapshot so that from_pretrained can find the files."""
    p = Path(model_path)
    if not p.is_dir():
        return model_path
    snapshots = p / "snapshots"
    if not snapshots.is_dir() or (p / "config.json").exists():
        return model_path  # Already a real model dir or not an HF cache wrapper
    snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return model_path
    resolved = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
    log.info("Resolved HF cache wrapper %s -> %s", model_path, resolved)
    return str(resolved)


def _client_endpoint_host(host: str) -> str:
    """Return a client-usable host for endpoints and health probes."""
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


class HostingService:
    """Manages model deployments as MCP servers or FastAPI endpoints."""

    def __init__(self):
        self._deployments: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _start_background_runtime(
        coro_factory: Callable[[], Awaitable[Any]],
        name: str,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "loop": None,
            "task": None,
            "thread": None,
            "error": None,
        }
        ready = threading.Event()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            state["loop"] = loop
            task = loop.create_task(coro_factory())
            state["task"] = task
            ready.set()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                state["error"] = exc
                log.exception("Background deployment runtime '%s' failed", name)
            finally:
                try:
                    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    for pending_task in pending:
                        pending_task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    log.debug("Background runtime '%s' shutdown cleanup failed", name, exc_info=True)
                finally:
                    loop.close()

        thread = threading.Thread(target=_runner, name=name, daemon=True)
        state["thread"] = thread
        thread.start()
        ready.wait(timeout=5.0)
        return state

    @staticmethod
    async def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
        target_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                if sock.connect_ex((target_host, port)) == 0:
                    return True
            await asyncio.sleep(0.2)
        return False

    async def deploy_as_mcp(self, config: HostingConfig) -> Dict[str, Any]:
        """Deploy a fine-tuned model as an MCP tool server."""
        try:
            from agentsoul.server import MCPServer, HTTPTransport, StdioTransport
            from agentsoul.providers.hf import HuggingFaceProvider

            resolved_model_path = _resolve_hf_cache_path(config.model_path)
            quant_arg = _quantization_to_provider_arg(config.quantization)

            gpu_lock = GPULock.get()
            await gpu_lock.acquire("deploy_mcp")
            try:
                provider = HuggingFaceProvider(
                    model_path=resolved_model_path,
                    lora_adapter_path=config.adapter_path,
                    **quant_arg,
                )
            finally:
                gpu_lock.release()

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

            runtime = self._start_background_runtime(
                lambda: transport.start(mcp),
                name=f"hosted-mcp-{config.port}",
            )
            if runtime.get("error") is not None:
                raise runtime["error"]

            if config.transport == "http":
                started = await self._wait_for_port(config.host, config.port)
                if not started:
                    task = runtime.get("task")
                    if task is not None and task.done():
                        try:
                            exc = task.exception()
                        except Exception:
                            exc = None
                        if exc is not None:
                            raise exc
                    raise TimeoutError(
                        f"Hosted MCP server did not start listening on port {config.port}"
                    )

            self._deployments[deployment_id] = {
                "id": deployment_id,
                "type": "mcp",
                "modality": config.modality,
                "model_path": config.model_path,
                "adapter_path": config.adapter_path,
                "transport": config.transport,
                "host": config.host,
                "port": config.port,
                "api_path": None,
                "routes": [],
                "task": runtime.get("task"),
                "loop": runtime.get("loop"),
                "thread": runtime.get("thread"),
                "mcp": mcp,
                "provider": provider,
            }

            return {
                "success": True,
                "deployment_id": deployment_id,
                "type": "mcp",
                "status": "running",
                "model_path": config.model_path,
                "modality": config.modality,
                "transport": config.transport,
                "endpoint": (
                    f"http://{_client_endpoint_host(config.host)}:{config.port}"
                    if config.transport == "http"
                    else "stdio"
                ),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def deploy_as_api(self, config: HostingConfig) -> Dict[str, Any]:
        """Deploy as a FastAPI endpoint with /generate POST route."""
        try:
            from fastapi import FastAPI
            import uvicorn
            from agentsoul.providers.hf import HuggingFaceProvider

            resolved_model_path = _resolve_hf_cache_path(config.model_path)
            quant_arg = _quantization_to_provider_arg(config.quantization)

            gpu_lock = GPULock.get()
            await gpu_lock.acquire("deploy_api")
            try:
                provider = HuggingFaceProvider(
                    model_path=resolved_model_path,
                    lora_adapter_path=config.adapter_path,
                    **quant_arg,
                )
            finally:
                gpu_lock.release()

            app = FastAPI(title=f"MCP Tuna Model: {config.model_path}")

            @app.post("/generate")
            async def generate(prompt: str, max_new_tokens: int = 512):
                resp = await provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_new_tokens=max_new_tokens,
                )
                return {"response": resp.content or ""}

            @app.get("/health")
            async def health():
                return {
                    "status": "ok",
                    "model": config.model_path,
                    "modality": config.modality,
                }

            deployment_id = str(uuid.uuid4())[:8]

            server_config = uvicorn.Config(app, host=config.host, port=config.port, log_level="info")
            server = uvicorn.Server(server_config)
            runtime = self._start_background_runtime(
                server.serve,
                name=f"hosted-api-{config.port}",
            )
            if runtime.get("error") is not None:
                raise runtime["error"]

            started = await self._wait_for_port(config.host, config.port)
            if not started:
                task = runtime.get("task")
                if task is not None and task.done():
                    try:
                        exc = task.exception()
                    except Exception:
                        exc = None
                    if exc is not None:
                        raise exc
                raise TimeoutError(
                    f"Hosted API server did not start listening on port {config.port}"
                )

            self._deployments[deployment_id] = {
                "id": deployment_id,
                "type": "api",
                "modality": config.modality,
                "model_path": config.model_path,
                "adapter_path": config.adapter_path,
                "transport": "http",
                "host": config.host,
                "port": config.port,
                "api_path": "/generate",
                "routes": ["/generate", "/health"],
                "task": runtime.get("task"),
                "loop": runtime.get("loop"),
                "thread": runtime.get("thread"),
                "server": server,
                "provider": provider,
            }

            return {
                "success": True,
                "deployment_id": deployment_id,
                "type": "api",
                "status": "running",
                "model_path": config.model_path,
                "modality": config.modality,
                "endpoint": f"http://{_client_endpoint_host(config.host)}:{config.port}",
                "routes": ["/generate", "/health"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def deploy_vlm_as_mcp(self, config: HostingConfig) -> Dict[str, Any]:
        """Deploy a VLM as an MCP tool server with a multimodal inference tool."""
        try:
            from agentsoul.server import MCPServer, HTTPTransport, StdioTransport

            from finetuning_pipeline.services.inference_service import InferenceService

            inference = InferenceService()
            mcp = MCPServer(f"hosted-vlm-{config.port}", "1.0.0")

            @mcp.tool(
                name="generate_vlm",
                description=f"Run multimodal generation with VLM at {config.model_path}",
            )
            async def generate_vlm(
                messages: List[Dict[str, Any]],
                max_new_tokens: int = 512,
            ) -> str:
                result = await inference.run_vlm_inference(
                    messages=messages,
                    model_path=config.model_path,
                    adapter_path=config.adapter_path,
                    max_new_tokens=max_new_tokens,
                )
                if not result.get("success"):
                    raise RuntimeError(result.get("error", "VLM generation failed"))
                return result.get("response", "")

            deployment_id = str(uuid.uuid4())[:8]

            if config.transport == "http":
                transport = HTTPTransport(host=config.host, port=config.port)
            else:
                transport = StdioTransport()

            runtime = self._start_background_runtime(
                lambda: transport.start(mcp),
                name=f"hosted-vlm-mcp-{config.port}",
            )
            if runtime.get("error") is not None:
                raise runtime["error"]

            if config.transport == "http":
                started = await self._wait_for_port(config.host, config.port)
                if not started:
                    task = runtime.get("task")
                    if task is not None and task.done():
                        try:
                            exc = task.exception()
                        except Exception:
                            exc = None
                        if exc is not None:
                            raise exc
                    raise TimeoutError(
                        f"Hosted VLM MCP server did not start listening on port {config.port}"
                    )

            self._deployments[deployment_id] = {
                "id": deployment_id,
                "type": "mcp",
                "modality": "vision-language",
                "model_path": config.model_path,
                "adapter_path": config.adapter_path,
                "transport": config.transport,
                "host": config.host,
                "port": config.port,
                "api_path": None,
                "routes": [],
                "task": runtime.get("task"),
                "loop": runtime.get("loop"),
                "thread": runtime.get("thread"),
                "mcp": mcp,
                "provider": None,
                "inference_service": inference,
            }

            return {
                "success": True,
                "deployment_id": deployment_id,
                "type": "mcp",
                "modality": "vision-language",
                "status": "running",
                "model_path": config.model_path,
                "transport": config.transport,
                "endpoint": (
                    f"http://{_client_endpoint_host(config.host)}:{config.port}"
                    if config.transport == "http"
                    else "stdio"
                ),
                "tools": ["generate_vlm"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def deploy_vlm_as_api(self, config: HostingConfig) -> Dict[str, Any]:
        """Deploy a VLM as a FastAPI endpoint with a /generate_vlm route."""
        try:
            from fastapi import FastAPI, HTTPException
            import uvicorn

            from finetuning_pipeline.services.inference_service import InferenceService

            inference = InferenceService()
            app = FastAPI(title=f"MCP Tuna VLM: {config.model_path}")

            @app.post("/generate_vlm")
            async def generate_vlm(payload: Dict[str, Any]):
                messages = payload.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise HTTPException(status_code=400, detail="Payload requires non-empty 'messages'")

                result = await inference.run_vlm_inference(
                    messages=messages,
                    model_path=config.model_path,
                    adapter_path=config.adapter_path,
                    max_new_tokens=int(payload.get("max_new_tokens", 512)),
                    temperature=float(payload.get("temperature", 0.7)),
                )
                if not result.get("success"):
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "VLM generation failed"),
                    )
                return {
                    "response": result.get("response", ""),
                    "details": result,
                }

            @app.get("/health")
            async def health():
                return {
                    "status": "ok",
                    "model": config.model_path,
                    "modality": "vision-language",
                }

            deployment_id = str(uuid.uuid4())[:8]

            server_config = uvicorn.Config(app, host=config.host, port=config.port, log_level="info")
            server = uvicorn.Server(server_config)
            runtime = self._start_background_runtime(
                server.serve,
                name=f"hosted-vlm-api-{config.port}",
            )
            if runtime.get("error") is not None:
                raise runtime["error"]

            started = await self._wait_for_port(config.host, config.port)
            if not started:
                task = runtime.get("task")
                if task is not None and task.done():
                    try:
                        exc = task.exception()
                    except Exception:
                        exc = None
                    if exc is not None:
                        raise exc
                raise TimeoutError(
                    f"Hosted VLM API server did not start listening on port {config.port}"
                )

            self._deployments[deployment_id] = {
                "id": deployment_id,
                "type": "api",
                "modality": "vision-language",
                "model_path": config.model_path,
                "adapter_path": config.adapter_path,
                "transport": "http",
                "host": config.host,
                "port": config.port,
                "api_path": "/generate_vlm",
                "routes": ["/generate_vlm", "/health"],
                "task": runtime.get("task"),
                "loop": runtime.get("loop"),
                "thread": runtime.get("thread"),
                "server": server,
                "provider": None,
                "inference_service": inference,
            }

            return {
                "success": True,
                "deployment_id": deployment_id,
                "type": "api",
                "modality": "vision-language",
                "status": "running",
                "model_path": config.model_path,
                "endpoint": f"http://{_client_endpoint_host(config.host)}:{config.port}",
                "routes": ["/generate_vlm", "/health"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _deployment_status(dep: Dict[str, Any]) -> str:
        thread = dep.get("thread")
        task = dep.get("task")
        if thread and thread.is_alive():
            return "running"
        if task is not None and not task.done():
            return "running"
        return "stopped"

    def get_deployment(self, deployment_id: str) -> Dict[str, Any] | None:
        """Return deployment metadata for a running deployment if present."""
        return self._deployments.get(deployment_id)

    async def list_deployments(self) -> Dict[str, Any]:
        """List currently running model deployments."""
        deployments = []
        for dep_id, dep in self._deployments.items():
            deployment = {
                "deployment_id": dep_id,
                "model_path": dep["model_path"],
                "adapter_path": dep.get("adapter_path"),
                "type": dep.get("type", "mcp"),
                "modality": dep.get("modality", "text"),
                "transport": dep["transport"],
                "status": self._deployment_status(dep),
                "endpoint": (
                    f"http://{_client_endpoint_host(dep['host'])}:{dep['port']}"
                    if dep["transport"] == "http"
                    else "stdio"
                ),
            }
            routes = dep.get("routes")
            if routes:
                deployment["routes"] = routes
            deployments.append(deployment)
        return {"success": True, "deployments": deployments, "count": len(deployments)}

    async def stop_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Stop a running deployment and free GPU memory."""
        if deployment_id not in self._deployments:
            return {"success": False, "error": f"Deployment {deployment_id} not found"}

        dep = self._deployments.pop(deployment_id)
        task = dep.get("task")
        server = dep.get("server")
        if server:
            server.should_exit = True

        loop = dep.get("loop")
        thread = dep.get("thread")
        if loop and task and not task.done():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        if thread and thread.is_alive():
            await asyncio.to_thread(thread.join, 5.0)

        provider = dep.get("provider")
        if provider and hasattr(provider, "unload"):
            try:
                provider.unload()
                log.info("Unloaded provider for deployment %s", deployment_id)
            except Exception:
                log.warning("Failed to unload provider for %s", deployment_id, exc_info=True)

        return {"success": True, "deployment_id": deployment_id, "status": "stopped"}

    async def health_check(self, deployment_id: str) -> Dict[str, Any]:
        """Check health of a running deployment."""
        if deployment_id not in self._deployments:
            return {"success": False, "error": f"Deployment {deployment_id} not found"}

        dep = self._deployments[deployment_id]
        thread = dep.get("thread")
        task = dep.get("task")
        is_alive = bool(thread and thread.is_alive())
        if not is_alive and task is not None and task.done():
            is_alive = False

        result: Dict[str, Any] = {
            "success": True,
            "deployment_id": deployment_id,
            "type": dep.get("type", "mcp"),
            "modality": dep.get("modality", "text"),
            "transport": dep.get("transport"),
            "status": "running" if is_alive else "stopped",
            "model_path": dep.get("model_path"),
            "adapter_path": dep.get("adapter_path"),
            "endpoint": (
                f"http://{_client_endpoint_host(dep['host'])}:{dep['port']}"
                if dep.get("transport") == "http" or dep.get("server")
                else "stdio"
            ),
        }
        if dep.get("routes"):
            result["routes"] = dep["routes"]

        # For API deployments, try hitting the /health endpoint
        if is_alive and dep.get("server"):
            try:
                import httpx
                probe_host = _client_endpoint_host(dep["host"])
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"http://{probe_host}:{dep['port']}/health")
                    result["health_response"] = resp.json()
            except Exception:
                result["health_response"] = None

        return result
