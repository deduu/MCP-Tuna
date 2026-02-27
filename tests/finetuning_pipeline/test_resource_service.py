"""Tests for ResourceService (system resource checking + VRAM preflight)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def resource_service():
    from finetuning_pipeline.services.resource_service import ResourceService
    return ResourceService()


# ------------------------------------------------------------------ #
# check_resources()
# ------------------------------------------------------------------ #


class TestCheckResources:
    """Tests for the hardware introspection method."""

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_returns_gpu_ram_disk_keys(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3050 Ti"
        props = MagicMock()
        props.total_memory = 4 * 1024**3  # 4 GB
        props.major = 8
        props.minor = 6
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0.8 * 1024**3
        mock_torch.cuda.memory_reserved.return_value = 1.0 * 1024**3
        mock_torch.version.cuda = "12.1"

        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mem.used = 8 * 1024**3
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.check_resources()
        assert "gpu" in result
        assert "ram" in result
        assert "disk" in result

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_gpu_available_true(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "RTX 3050 Ti"
        props = MagicMock()
        props.total_memory = 4 * 1024**3
        props.major = 8
        props.minor = 6
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0
        mock_torch.version.cuda = "12.1"

        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mem.used = 8 * 1024**3
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.check_resources()
        assert result["gpu"]["available"] is True
        assert result["gpu"]["name"] == "RTX 3050 Ti"
        assert result["gpu"]["vram_total_gb"] == pytest.approx(4.0, abs=0.1)

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_no_gpu(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = False

        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mem.used = 8 * 1024**3
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.check_resources()
        assert result["gpu"]["available"] is False
        assert result["ram"]["total_gb"] > 0

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_ram_values(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = False

        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mem.used = 8 * 1024**3
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.check_resources()
        assert result["ram"]["total_gb"] == pytest.approx(16.0, abs=0.1)
        assert result["ram"]["free_gb"] == pytest.approx(8.0, abs=0.1)
        assert result["ram"]["percent_used"] == pytest.approx(50.0, abs=0.1)


# ------------------------------------------------------------------ #
# preflight_check()
# ------------------------------------------------------------------ #


class TestPreflightCheck:
    """Tests for VRAM estimation and feasibility checking."""

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_1b_4bit_fits_in_4gb(self, mock_torch, mock_psutil, resource_service):
        """Llama 1B 4-bit should easily fit in 4GB VRAM."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 4 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            quantization="4bit",
            batch_size=1,
            max_seq_length=512,
        )
        assert result["can_run"] is True
        assert result["estimated_vram_gb"] < 4.0

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_70b_4bit_does_not_fit_in_4gb(self, mock_torch, mock_psutil, resource_service):
        """Llama 70B even at 4-bit won't fit in 4GB."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 4 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            quantization="4bit",
            batch_size=1,
            max_seq_length=512,
        )
        assert result["can_run"] is False
        assert result["estimated_vram_gb"] > 4.0

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_3b_4bit_fits_in_4gb(self, mock_torch, mock_psutil, resource_service):
        """Llama 3B 4-bit should fit in 4GB with conservative settings."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 4 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
            batch_size=1,
            max_seq_length=256,
            gradient_checkpointing=True,
        )
        assert result["can_run"] is True
        assert result["estimated_vram_gb"] < 4.0

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_gradient_checkpointing_reduces_activation_estimate(
        self, mock_torch, mock_psutil, resource_service
    ):
        """gradient_checkpointing=True should lower activation memory."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        without_gc = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
            batch_size=1,
            max_seq_length=512,
            gradient_checkpointing=False,
        )
        with_gc = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
            batch_size=1,
            max_seq_length=512,
            gradient_checkpointing=True,
        )
        assert with_gc["breakdown"]["activations_gb"] < without_gc["breakdown"]["activations_gb"]
        assert with_gc["estimated_vram_gb"] < without_gc["estimated_vram_gb"]

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_no_gpu_cannot_run(self, mock_torch, mock_psutil, resource_service):
        """Without GPU, preflight should return can_run=False."""
        mock_torch.cuda.is_available.return_value = False

        result = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            quantization="4bit",
        )
        assert result["can_run"] is False
        assert any("no cuda gpu" in w.lower() for w in result["warnings"])

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_breakdown_has_required_keys(self, mock_torch, mock_psutil, resource_service):
        """The VRAM breakdown should contain all expected components."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
        )
        breakdown = result["breakdown"]
        assert "model_gb" in breakdown
        assert "lora_gb" in breakdown
        assert "optimizer_gb" in breakdown
        assert "activations_gb" in breakdown
        assert "overhead_gb" in breakdown

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_breakdown_sums_to_estimated_total(self, mock_torch, mock_psutil, resource_service):
        """The breakdown components should sum to the estimated total VRAM."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
        )
        breakdown = result["breakdown"]
        component_sum = sum(breakdown.values())
        assert result["estimated_vram_gb"] == pytest.approx(component_sum, abs=0.01)

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_unknown_model_uses_fallback(self, mock_torch, mock_psutil, resource_service):
        """Unknown models should still return an estimate (with a warning)."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 24 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result = resource_service.preflight_check(
            model_name="some-org/unknown-model-7B",
            quantization="4bit",
        )
        # Should still produce a result (with warning about estimated params)
        assert "estimated_vram_gb" in result
        assert "breakdown" in result

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_no_lora_increases_optimizer_memory(self, mock_torch, mock_psutil, resource_service):
        """Full fine-tuning (no LoRA) should have higher optimizer cost."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 48 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        with_lora = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
            use_lora=True,
        )
        without_lora = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
            use_lora=False,
        )
        # Without LoRA, optimizer memory should be higher (full param updates)
        assert without_lora["breakdown"]["optimizer_gb"] > with_lora["breakdown"]["optimizer_gb"]

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_recommendations_present(self, mock_torch, mock_psutil, resource_service):
        """Should include recommendations when applicable."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 4 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
            gradient_checkpointing=False,
        )
        assert isinstance(result["recommendations"], list)

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_fp16_quantization(self, mock_torch, mock_psutil, resource_service):
        """FP16 should require significantly more VRAM than 4-bit."""
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 48 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        result_4bit = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="4bit",
        )
        result_fp16 = resource_service.preflight_check(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            quantization="none",
        )
        assert result_fp16["breakdown"]["model_gb"] > result_4bit["breakdown"]["model_gb"]
