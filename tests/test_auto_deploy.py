"""Tests for auto-deploy on individual training tools (Feature 2)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock



# ----------------------------------------------------------------
# _auto_deploy_if_requested helper tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_deploy_called_when_deploy_true_and_success():
    """deploy=True + successful training → deploy_as_mcp should be called."""
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._finetuning_svc = MagicMock()
    gw._finetuning_svc.config = MagicMock()
    gw._finetuning_svc.config.base_model = "meta-llama/Llama-3.2-3B-Instruct"
    gw._hosting_svc = AsyncMock()
    gw._hosting_svc.deploy_as_mcp = AsyncMock(
        return_value={"success": True, "deployment_id": "abc123"}
    )

    train_result = {"success": True, "model_path": "/output/sft_v1"}
    deploy_result = await gw._auto_deploy_if_requested(
        train_result, deploy=True, deploy_port=8001, base_model=None
    )

    assert deploy_result is not None
    assert deploy_result["success"] is True
    gw._hosting_svc.deploy_as_mcp.assert_called_once()

    # Verify HostingConfig was correct
    call_args = gw._hosting_svc.deploy_as_mcp.call_args
    config = call_args.args[0] if call_args.args else call_args.kwargs.get("config")
    assert config.adapter_path == "/output/sft_v1"
    assert config.port == 8001


@pytest.mark.asyncio
async def test_auto_deploy_not_called_when_deploy_false():
    """deploy=False → deploy_as_mcp should never be called."""
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._hosting_svc = AsyncMock()

    train_result = {"success": True, "model_path": "/output/sft_v1"}
    deploy_result = await gw._auto_deploy_if_requested(
        train_result, deploy=False, deploy_port=8001
    )

    assert deploy_result is None
    gw._hosting_svc.deploy_as_mcp.assert_not_called()


@pytest.mark.asyncio
async def test_auto_deploy_not_called_on_training_failure():
    """Training failure → no deploy attempt even if deploy=True."""
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._hosting_svc = AsyncMock()

    train_result = {"success": False, "error": "OOM"}
    deploy_result = await gw._auto_deploy_if_requested(
        train_result, deploy=True, deploy_port=8001
    )

    assert deploy_result is None


@pytest.mark.asyncio
async def test_auto_deploy_uses_correct_port():
    """deploy_port should be passed to HostingConfig."""
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._finetuning_svc = MagicMock()
    gw._finetuning_svc.config.base_model = "fake-base"
    gw._hosting_svc = AsyncMock()
    gw._hosting_svc.deploy_as_mcp = AsyncMock(return_value={"success": True})

    train_result = {"success": True, "model_path": "/model"}
    await gw._auto_deploy_if_requested(
        train_result, deploy=True, deploy_port=9999
    )

    config = gw._hosting_svc.deploy_as_mcp.call_args.args[0]
    assert config.port == 9999


@pytest.mark.asyncio
async def test_auto_deploy_uses_explicit_base_model():
    """When base_model is provided, it should be used as model_path in HostingConfig."""
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._finetuning_svc = MagicMock()
    gw._finetuning_svc.config.base_model = "default-base"
    gw._hosting_svc = AsyncMock()
    gw._hosting_svc.deploy_as_mcp = AsyncMock(return_value={"success": True})

    train_result = {"success": True, "model_path": "/adapter"}
    await gw._auto_deploy_if_requested(
        train_result, deploy=True, deploy_port=8001, base_model="custom-base"
    )

    config = gw._hosting_svc.deploy_as_mcp.call_args.args[0]
    assert config.model_path == "custom-base"
    assert config.adapter_path == "/adapter"


@pytest.mark.asyncio
async def test_auto_deploy_handles_final_model_path():
    """For curriculum/sequential results using final_model_path instead of model_path."""
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._finetuning_svc = MagicMock()
    gw._finetuning_svc.config.base_model = "fake-base"
    gw._hosting_svc = AsyncMock()
    gw._hosting_svc.deploy_as_mcp = AsyncMock(return_value={"success": True})

    train_result = {"success": True, "final_model_path": "/output/curriculum/stage_3"}
    await gw._auto_deploy_if_requested(
        train_result, deploy=True, deploy_port=8001
    )

    config = gw._hosting_svc.deploy_as_mcp.call_args.args[0]
    assert config.adapter_path == "/output/curriculum/stage_3"


@pytest.mark.asyncio
async def test_auto_deploy_uses_trained_model_directly_when_lora_disabled():
    """Non-LoRA training outputs should deploy without adapter_path."""
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._finetuning_svc = MagicMock()
    gw._finetuning_svc.config.base_model = "fake-base"
    gw._hosting_svc = AsyncMock()
    gw._hosting_svc.deploy_as_mcp = AsyncMock(return_value={"success": True})

    train_result = {
        "success": True,
        "model_path": "/output/full-model",
        "config": {"trainer": "sft", "use_lora": False},
    }
    await gw._auto_deploy_if_requested(
        train_result, deploy=True, deploy_port=8001
    )

    config = gw._hosting_svc.deploy_as_mcp.call_args.args[0]
    assert config.model_path == "/output/full-model"
    assert config.adapter_path is None


@pytest.mark.asyncio
async def test_auto_deploy_uses_vlm_deployer_for_vlm_training():
    from mcp_gateway import TunaGateway

    gw = TunaGateway.__new__(TunaGateway)
    gw._finetuning_svc = MagicMock()
    gw._finetuning_svc.config.base_model = "fake-vlm-base"
    gw._hosting_svc = AsyncMock()
    gw._hosting_svc.deploy_as_mcp = AsyncMock(return_value={"success": True})
    gw._hosting_svc.deploy_vlm_as_mcp = AsyncMock(return_value={"success": True})

    train_result = {
        "success": True,
        "model_path": "/output/vlm-adapter",
        "config": {"trainer": "vlm_sft", "use_lora": True},
    }
    await gw._auto_deploy_if_requested(
        train_result, deploy=True, deploy_port=8101, base_model="Qwen/Qwen2.5-VL-3B-Instruct"
    )

    gw._hosting_svc.deploy_vlm_as_mcp.assert_called_once()
    gw._hosting_svc.deploy_as_mcp.assert_not_called()
    config = gw._hosting_svc.deploy_vlm_as_mcp.call_args.args[0]
    assert config.modality == "vision-language"
    assert config.model_path == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config.adapter_path == "/output/vlm-adapter"
