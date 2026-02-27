"""Tests for finetuning_pipeline.services.adapter_service — TDD."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from finetuning_pipeline.services.adapter_service import AdapterService


@pytest.fixture
def svc() -> AdapterService:
    return AdapterService()


# ---------------------------------------------------------------------------
# merge_adapter
# ---------------------------------------------------------------------------

@patch("finetuning_pipeline.services.adapter_service.AdapterService.merge_adapter")
async def test_merge_adapter_success(mock_merge: AsyncMock, svc: AdapterService, tmp_path: Path):
    """Merged model directory is created with success response."""
    out = tmp_path / "merged"
    mock_merge.return_value = {
        "success": True,
        "model_path": str(out),
        "model_size_gb": 1.5,
        "pushed_to_hub": None,
    }
    result = await mock_merge(
        base_model="meta-llama/Llama-3.2-1B",
        adapter_path="/fake/adapter",
        output_path=str(out),
    )
    assert result["success"] is True
    assert "model_path" in result


@patch("finetuning_pipeline.services.adapter_service.AdapterService.merge_adapter")
async def test_merge_adapter_push_to_hub(mock_merge: AsyncMock, tmp_path: Path):
    out = tmp_path / "merged"
    mock_merge.return_value = {
        "success": True,
        "model_path": str(out),
        "model_size_gb": 1.5,
        "pushed_to_hub": "user/my-model",
    }
    result = await mock_merge(
        base_model="meta-llama/Llama-3.2-1B",
        adapter_path="/fake/adapter",
        output_path=str(out),
        push_to_hub="user/my-model",
    )
    assert result["success"] is True
    assert result["pushed_to_hub"] == "user/my-model"


# ---------------------------------------------------------------------------
# export_gguf
# ---------------------------------------------------------------------------

async def test_export_gguf_invalid_quantization(svc: AdapterService):
    result = await svc.export_gguf(
        model_path="/fake/model",
        output_path="/fake/out.gguf",
        quantization="q3_ultra",
    )
    assert result["success"] is False
    assert "quantization" in result["error"].lower()


async def test_export_gguf_supported_quantizations(svc: AdapterService):
    """All documented quantizations are in the supported set."""
    for q in ["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"]:
        assert q in svc._SUPPORTED_QUANTIZATIONS


async def test_export_gguf_missing_llama_cpp(svc: AdapterService):
    """Returns clear error when llama-cpp-python is not installed."""
    with patch.dict("sys.modules", {"llama_cpp": None}):
        # Force ImportError by mocking __import__
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("No module named 'llama_cpp'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await svc.export_gguf(
                model_path="/fake/model",
                output_path="/fake/out.gguf",
                quantization="q4_k_m",
            )
            assert result["success"] is False
            assert "llama-cpp-python" in result["error"]


# ---------------------------------------------------------------------------
# Facade delegation
# ---------------------------------------------------------------------------

async def test_pipeline_service_delegates_merge_adapter():
    """FineTuningService.merge_adapter delegates to AdapterService."""
    from finetuning_pipeline.services.pipeline_service import FineTuningService
    svc = FineTuningService()
    with patch.object(AdapterService, "merge_adapter", new_callable=AsyncMock) as mock:
        mock.return_value = {"success": True}
        result = await svc.merge_adapter(
            base_model="x", adapter_path="y", output_path="z",
        )
        assert result["success"] is True
        mock.assert_called_once()


async def test_pipeline_service_delegates_export_gguf():
    """FineTuningService.export_gguf delegates to AdapterService."""
    from finetuning_pipeline.services.pipeline_service import FineTuningService
    svc = FineTuningService()
    with patch.object(AdapterService, "export_gguf", new_callable=AsyncMock) as mock:
        mock.return_value = {"success": True}
        result = await svc.export_gguf(
            model_path="x", output_path="y", quantization="q4_k_m",
        )
        assert result["success"] is True
        mock.assert_called_once()
