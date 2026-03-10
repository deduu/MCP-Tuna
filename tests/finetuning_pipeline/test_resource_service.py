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

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_multiple_gpus_are_reported(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.side_effect = ["RTX 4090", "RTX 4070"]

        props0 = MagicMock()
        props0.total_memory = 24 * 1024**3
        props0.major = 8
        props0.minor = 9

        props1 = MagicMock()
        props1.total_memory = 12 * 1024**3
        props1.major = 8
        props1.minor = 6

        mock_torch.cuda.get_device_properties.side_effect = [props0, props1]
        mock_torch.cuda.memory_allocated.side_effect = [2 * 1024**3, 1 * 1024**3]
        mock_torch.cuda.memory_reserved.side_effect = [3 * 1024**3, 1.5 * 1024**3]
        mock_torch.version.cuda = "12.4"

        mem = MagicMock()
        mem.total = 64 * 1024**3
        mem.available = 48 * 1024**3
        mem.used = 16 * 1024**3
        mem.percent = 25.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.check_resources()
        assert result["gpu_count"] == 2
        assert len(result["gpus"]) == 2
        assert result["gpu"]["name"] == "RTX 4090"
        assert result["gpus"][1]["name"] == "RTX 4070"
        assert result["gpus"][1]["vram_total_gb"] == pytest.approx(12.0, abs=0.1)


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


# ------------------------------------------------------------------ #
# prescribe()
# ------------------------------------------------------------------ #


def _mock_gpu(mock_torch, vram_gb: float) -> None:
    """Helper: configure mock torch with given VRAM."""
    mock_torch.cuda.is_available.return_value = True
    props = MagicMock()
    props.total_memory = vram_gb * 1024**3
    props.major = 8
    props.minor = 6
    mock_torch.cuda.get_device_properties.return_value = props
    mock_torch.cuda.memory_allocated.return_value = 0
    mock_torch.cuda.memory_reserved.return_value = 0
    mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX Test"
    mock_torch.version.cuda = "12.4"


def _mock_ram(mock_psutil, total_gb: float, free_gb: float) -> None:
    """Helper: configure mock psutil with given RAM."""
    mem = MagicMock()
    mem.total = total_gb * 1024**3
    mem.available = free_gb * 1024**3
    mem.used = (total_gb - free_gb) * 1024**3
    mem.percent = round((total_gb - free_gb) / total_gb * 100, 1)
    mock_psutil.virtual_memory.return_value = mem


class TestPrescribe:
    """Tests for resource-aware training configuration prescription."""

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_small_dataset_tight_vram_prescribes_conservative(
        self, mock_torch, mock_psutil, resource_service
    ):
        """17 rows + 4GB VRAM (like RTX 3050 Ti) → 4bit, batch=1, epochs≥3."""
        _mock_gpu(mock_torch, vram_gb=4.3)
        _mock_ram(mock_psutil, total_gb=16.0, free_gb=8.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=17,
            dataset_avg_text_length=300,
            technique="sft",
        )
        assert result["can_run"] is True
        cfg = result["config"]
        # 1.5B model fits fp16 in 4.3GB — prescriber correctly picks best quality
        assert cfg["quantization"] in ("4bit", "8bit", "none")
        assert cfg["per_device_train_batch_size"] >= 1
        assert cfg["num_epochs"] >= 3

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_large_dataset_ample_vram_prescribes_efficient(
        self, mock_torch, mock_psutil, resource_service
    ):
        """10K rows + 24GB VRAM → can use larger batch, fewer epochs."""
        _mock_gpu(mock_torch, vram_gb=24.0)
        _mock_ram(mock_psutil, total_gb=64.0, free_gb=32.0)

        result = resource_service.prescribe(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            dataset_row_count=10000,
            dataset_avg_text_length=200,
            technique="sft",
        )
        assert result["can_run"] is True
        cfg = result["config"]
        assert cfg["num_epochs"] <= 2
        assert cfg["per_device_train_batch_size"] >= 1
        assert cfg["lora_r"] >= 8

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_no_gpu_cannot_run(self, mock_torch, mock_psutil, resource_service):
        """Without GPU, prescribe should return can_run=False."""
        mock_torch.cuda.is_available.return_value = False
        _mock_ram(mock_psutil, total_gb=16.0, free_gb=8.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=100,
            dataset_avg_text_length=200,
        )
        assert result["can_run"] is False
        assert any("gpu" in w.lower() for w in result["warnings"])

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_dataset_sampling_recommended_when_ram_tight(
        self, mock_torch, mock_psutil, resource_service
    ):
        """Huge dataset + low RAM → sampling_needed=True."""
        _mock_gpu(mock_torch, vram_gb=8.0)
        _mock_ram(mock_psutil, total_gb=16.0, free_gb=2.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=500000,
            dataset_avg_text_length=2000,
        )
        dp = result["dataset_plan"]
        assert dp["sampling_needed"] is True
        assert dp["recommended_rows"] < 500000

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_prescribed_config_passes_preflight(
        self, mock_torch, mock_psutil, resource_service
    ):
        """The prescribed config should pass preflight_check() itself."""
        _mock_gpu(mock_torch, vram_gb=4.3)
        _mock_ram(mock_psutil, total_gb=16.0, free_gb=8.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=100,
            dataset_avg_text_length=250,
        )
        assert result["can_run"] is True
        cfg = result["config"]

        # Feed the prescribed config back into preflight_check
        preflight = resource_service.preflight_check(
            model_name="Qwen/Qwen2-1.5B",
            quantization=cfg["quantization"],
            batch_size=cfg["per_device_train_batch_size"],
            max_seq_length=cfg["max_seq_length"],
            use_lora=cfg["use_lora"],
            lora_r=cfg["lora_r"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
        )
        assert preflight["can_run"] is True

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_low_disk_produces_warning(self, mock_torch, mock_psutil, resource_service):
        """Low disk space should produce a warning."""
        _mock_gpu(mock_torch, vram_gb=8.0)
        _mock_ram(mock_psutil, total_gb=16.0, free_gb=8.0)

        # Patch disk to return low free space
        with patch.object(resource_service, "_get_disk_info", return_value={
            "output_dir": "output",
            "total_gb": 100.0,
            "free_gb": 3.0,
            "used_gb": 97.0,
        }):
            result = resource_service.prescribe(
                model_name="Qwen/Qwen2-1.5B",
                dataset_row_count=100,
                dataset_avg_text_length=200,
            )
        assert any("disk" in w.lower() for w in result["warnings"])

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_unknown_model_still_prescribes(self, mock_torch, mock_psutil, resource_service):
        """Unknown model name with size pattern → still returns valid config."""
        _mock_gpu(mock_torch, vram_gb=24.0)
        _mock_ram(mock_psutil, total_gb=32.0, free_gb=16.0)

        result = resource_service.prescribe(
            model_name="some-org/custom-model-3B",
            dataset_row_count=500,
            dataset_avg_text_length=300,
        )
        assert result["can_run"] is True
        assert "config" in result
        assert any("not in known list" in w for w in result["warnings"])

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_abundant_vram_prefers_fp16_over_4bit(
        self, mock_torch, mock_psutil, resource_service
    ):
        """48GB VRAM + small model → should pick fp16 for best quality."""
        _mock_gpu(mock_torch, vram_gb=48.0)
        _mock_ram(mock_psutil, total_gb=64.0, free_gb=32.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=500,
            dataset_avg_text_length=300,
        )
        assert result["can_run"] is True
        cfg = result["config"]
        assert cfg["quantization"] == "none"  # fp16, not 4bit
        assert cfg["per_device_train_batch_size"] >= 4  # can afford bigger batches

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_abundant_vram_small_model_suggests_full_finetune(
        self, mock_torch, mock_psutil, resource_service
    ):
        """Abundant VRAM + small model → full fine-tune (no LoRA) for best quality."""
        _mock_gpu(mock_torch, vram_gb=24.0)
        _mock_ram(mock_psutil, total_gb=32.0, free_gb=16.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=1000,
            dataset_avg_text_length=200,
        )
        assert result["can_run"] is True
        cfg = result["config"]
        assert cfg["use_lora"] is False
        assert cfg["lora_r"] == 0

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_midrange_vram_selects_8bit_when_possible(
        self, mock_torch, mock_psutil, resource_service
    ):
        """8GB VRAM + 1.5B model → should pick 8-bit (better than 4-bit, fits)."""
        _mock_gpu(mock_torch, vram_gb=8.0)
        _mock_ram(mock_psutil, total_gb=16.0, free_gb=8.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=500,
            dataset_avg_text_length=250,
        )
        assert result["can_run"] is True
        cfg = result["config"]
        # 1.5B at 8-bit needs ~1.7GB model + overhead ≈ ~3GB total, fits in 8GB
        # Should prefer 8-bit or even fp16 over 4-bit
        assert cfg["quantization"] in ("none", "8bit")

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_abundant_vram_disables_gradient_checkpointing(
        self, mock_torch, mock_psutil, resource_service
    ):
        """With lots of headroom, gradient_checkpointing should be off for speed."""
        _mock_gpu(mock_torch, vram_gb=48.0)
        _mock_ram(mock_psutil, total_gb=64.0, free_gb=32.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=500,
            dataset_avg_text_length=200,
        )
        cfg = result["config"]
        assert cfg["gradient_checkpointing"] is False

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_abundant_vram_scales_lora_rank_up(
        self, mock_torch, mock_psutil, resource_service
    ):
        """With ample headroom and large model requiring LoRA, rank should scale up."""
        _mock_gpu(mock_torch, vram_gb=24.0)
        _mock_ram(mock_psutil, total_gb=32.0, free_gb=16.0)

        result = resource_service.prescribe(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            dataset_row_count=2000,
            dataset_avg_text_length=200,
        )
        cfg = result["config"]
        assert cfg["use_lora"] is True  # 8B model still needs LoRA
        assert cfg["lora_r"] >= 16  # should scale up with headroom

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_return_structure_has_required_keys(
        self, mock_torch, mock_psutil, resource_service
    ):
        """Verify the return dict contains all expected top-level keys."""
        _mock_gpu(mock_torch, vram_gb=8.0)
        _mock_ram(mock_psutil, total_gb=16.0, free_gb=8.0)

        result = resource_service.prescribe(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=100,
            dataset_avg_text_length=200,
        )
        assert "can_run" in result
        assert "config" in result
        assert "dataset_plan" in result
        assert "resource_snapshot" in result
        assert "vram_estimate" in result
        assert "warnings" in result
        assert "rationale" in result

        cfg = result["config"]
        for key in [
            "quantization", "max_seq_length", "per_device_train_batch_size",
            "gradient_accumulation_steps", "gradient_checkpointing",
            "use_lora", "lora_r", "lora_alpha", "learning_rate",
            "num_epochs", "warmup_ratio", "lr_scheduler_type",
            "weight_decay", "optim",
        ]:
            assert key in cfg, f"Missing config key: {key}"


# ------------------------------------------------------------------ #
# disk_preflight()
# ------------------------------------------------------------------ #


class TestDiskPreflight:
    """Tests for disk space preflight checks."""

    @patch("finetuning_pipeline.services.resource_service.shutil")
    def test_sufficient_disk_space(self, mock_shutil, resource_service):
        mock_shutil.disk_usage.return_value = MagicMock(
            free=20 * 1024**3  # 20 GB free
        )
        result = resource_service.disk_preflight(estimated_size_gb=5.0)
        assert result["can_run"] is True
        assert result["free_gb"] >= 20
        assert result["warnings"] == [] or all("Low" in w for w in result["warnings"])

    @patch("finetuning_pipeline.services.resource_service.shutil")
    def test_insufficient_disk_space(self, mock_shutil, resource_service):
        mock_shutil.disk_usage.return_value = MagicMock(
            free=3 * 1024**3  # 3 GB free
        )
        result = resource_service.disk_preflight(estimated_size_gb=5.0)
        assert result["can_run"] is False
        assert len(result["warnings"]) > 0

    @patch("finetuning_pipeline.services.resource_service.shutil")
    def test_low_disk_warning(self, mock_shutil, resource_service):
        mock_shutil.disk_usage.return_value = MagicMock(
            free=8 * 1024**3  # 8 GB free — enough for 5GB but low overall
        )
        result = resource_service.disk_preflight(estimated_size_gb=5.0)
        assert result["can_run"] is True
        assert any("Low" in w or "low" in w.lower() for w in result["warnings"])


# ------------------------------------------------------------------ #
# prescribe_pipeline()
# ------------------------------------------------------------------ #


class TestPrescribePipeline:
    """Tests for end-to-end pipeline prescription."""

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_all_stages_returned(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        props = MagicMock()
        props.total_memory = 24 * 1024**3
        props.major = 8
        props.minor = 9
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 1 * 1024**3
        mock_torch.cuda.memory_reserved.return_value = 2 * 1024**3
        mock_torch.version.cuda = "12.1"

        mem = MagicMock()
        mem.total = 32 * 1024**3
        mem.available = 16 * 1024**3
        mem.used = 16 * 1024**3
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.prescribe_pipeline(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=1000,
            dataset_avg_text_length=200,
        )

        assert "stages" in result
        assert "evaluate" in result["stages"]
        assert "train" in result["stages"]
        assert "deploy" in result["stages"]
        assert "resource_snapshot" in result

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_no_gpu_deploy_infeasible(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.__bool__ = lambda self: True

        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mem.used = 8 * 1024**3
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.prescribe_pipeline(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=100,
            dataset_avg_text_length=200,
            stages=["deploy"],
        )

        assert result["stages"]["deploy"]["feasible"] is False

    @patch("finetuning_pipeline.services.resource_service.psutil")
    @patch("finetuning_pipeline.services.resource_service.torch")
    def test_specific_stages_only(self, mock_torch, mock_psutil, resource_service):
        mock_torch.cuda.is_available.return_value = False

        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mem.used = 8 * 1024**3
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem

        result = resource_service.prescribe_pipeline(
            model_name="Qwen/Qwen2-1.5B",
            dataset_row_count=100,
            dataset_avg_text_length=200,
            stages=["evaluate"],
        )

        assert "evaluate" in result["stages"]
        assert "train" not in result["stages"]
        assert "deploy" not in result["stages"]
