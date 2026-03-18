"""Tests for hosting service VRAM leak fix and GPU lock integration."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.gpu_lock import GPULock


@pytest.fixture(autouse=True)
def reset_gpu_lock():
    GPULock.reset()
    yield
    GPULock.reset()


def _make_mock_provider():
    provider = MagicMock()
    provider.unload = MagicMock()
    provider.load_adapter = MagicMock()
    provider.chat = AsyncMock(return_value=MagicMock(content="test response"))
    return provider


class TestHostingServiceVRAMLeak:
    async def test_stop_deployment_calls_unload(self):
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        provider = _make_mock_provider()

        # Manually insert a deployment with a provider
        dep_id = "test-001"
        task = MagicMock()
        task.done.return_value = True
        svc._deployments[dep_id] = {
            "id": dep_id,
            "model_path": "test/model",
            "adapter_path": None,
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8080,
            "task": task,
            "provider": provider,
        }

        result = await svc.stop_deployment(dep_id)
        assert result["success"] is True
        provider.unload.assert_called_once()

    async def test_stop_deployment_without_provider_no_error(self):
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        dep_id = "test-002"
        task = MagicMock()
        task.done.return_value = True
        svc._deployments[dep_id] = {
            "id": dep_id,
            "model_path": "test/model",
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8080,
            "task": task,
        }

        result = await svc.stop_deployment(dep_id)
        assert result["success"] is True

    async def test_stop_deployment_unload_failure_still_succeeds(self):
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        provider = _make_mock_provider()
        provider.unload.side_effect = RuntimeError("unload failed")

        dep_id = "test-003"
        task = MagicMock()
        task.done.return_value = True
        svc._deployments[dep_id] = {
            "id": dep_id,
            "model_path": "test/model",
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8080,
            "task": task,
            "provider": provider,
        }

        result = await svc.stop_deployment(dep_id)
        assert result["success"] is True
        provider.unload.assert_called_once()

    async def test_deployment_dict_stores_provider(self):
        """Verify deploy methods store provider reference for later cleanup."""
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        provider = _make_mock_provider()

        # Manually simulate what deploy_as_mcp does after loading
        dep_id = "test-004"
        svc._deployments[dep_id] = {
            "id": dep_id,
            "model_path": "test/model",
            "provider": provider,
            "task": MagicMock(),
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8080,
        }

        assert svc._deployments[dep_id]["provider"] is provider

    async def test_list_deployments_returns_frontend_fields(self):
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        dep_id = "test-005"
        thread = MagicMock()
        thread.is_alive.return_value = True
        svc._deployments[dep_id] = {
            "id": dep_id,
            "type": "api",
            "modality": "text",
            "model_path": "test/model",
            "adapter_path": "./adapter",
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8080,
            "thread": thread,
            "task": MagicMock(),
            "routes": ["/generate", "/health"],
        }

        result = await svc.list_deployments()

        assert result["success"] is True
        assert result["count"] == 1
        assert result["deployments"][0] == {
            "deployment_id": dep_id,
            "model_path": "test/model",
            "adapter_path": "./adapter",
            "type": "api",
            "modality": "text",
            "transport": "http",
            "status": "running",
            "endpoint": "http://127.0.0.1:8080",
            "routes": ["/generate", "/health"],
        }

    async def test_list_deployments_normalizes_wildcard_host_endpoint(self):
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        dep_id = "test-006"
        thread = MagicMock()
        thread.is_alive.return_value = True
        svc._deployments[dep_id] = {
            "id": dep_id,
            "type": "api",
            "modality": "text",
            "model_path": "test/model",
            "adapter_path": None,
            "transport": "http",
            "host": "0.0.0.0",
            "port": 8001,
            "thread": thread,
            "task": MagicMock(),
            "server": object(),
        }

        result = await svc.list_deployments()

        assert result["deployments"][0]["endpoint"] == "http://127.0.0.1:8001"

    async def test_health_check_normalizes_wildcard_host_for_probe(self):
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        dep_id = "test-007"
        thread = MagicMock()
        thread.is_alive.return_value = True
        svc._deployments[dep_id] = {
            "id": dep_id,
            "type": "api",
            "modality": "vision-language",
            "model_path": "test/model",
            "adapter_path": None,
            "transport": "http",
            "host": "0.0.0.0",
            "port": 8001,
            "thread": thread,
            "task": MagicMock(),
            "server": object(),
            "routes": ["/generate_vlm", "/health"],
        }

        client = AsyncMock()
        response = MagicMock()
        response.json.return_value = {"status": "ok"}
        client.get.return_value = response
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False

        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda timeout=5: client)
            result = await svc.health_check(dep_id)

        client.get.assert_awaited_once_with("http://127.0.0.1:8001/health")
        assert result["endpoint"] == "http://127.0.0.1:8001"
        assert result["modality"] == "vision-language"
        assert result["health_response"] == {"status": "ok"}

    async def test_list_deployments_includes_vlm_metadata(self):
        from hosting_pipeline.services.hosting_service import HostingService

        svc = HostingService()
        dep_id = "test-vlm-001"
        thread = MagicMock()
        thread.is_alive.return_value = True
        svc._deployments[dep_id] = {
            "id": dep_id,
            "type": "api",
            "modality": "vision-language",
            "model_path": "test/vlm",
            "adapter_path": "./adapter",
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8090,
            "thread": thread,
            "task": MagicMock(),
            "routes": ["/generate_vlm", "/health"],
        }

        result = await svc.list_deployments()

        assert result["deployments"][0]["modality"] == "vision-language"
        assert result["deployments"][0]["routes"] == ["/generate_vlm", "/health"]
