"""Unit tests for SequentialTrainingService (Feature 1)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _mock_load_result(dataset_object=None):
    """Return a successful load_dataset_from_file result."""
    return {
        "success": True,
        "dataset_object": dataset_object or [{"instruction": "q", "output": "a"}],
        "num_examples": 1,
    }


def _mock_train_result(output_dir: str):
    """Return a successful training result."""
    return {
        "success": True,
        "model_path": output_dir,
        "base_model": "fake-base",
        "num_training_examples": 1,
    }


def _mock_train_failure(msg: str = "OOM"):
    return {"success": False, "error": msg}


# ----------------------------------------------------------------
# Validation tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_stages_returns_error():
    """Empty stages list should return an error immediately."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    svc = SequentialTrainingService()
    result = await svc.train_sequential(stages=[], output_dir="/tmp/seq")
    assert result["success"] is False
    assert "No stages" in result["error"]


@pytest.mark.asyncio
async def test_invalid_technique_returns_error():
    """Unknown technique name should fail with a clear message."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[{"technique": "reinforce", "dataset_path": "/data.jsonl"}],
        output_dir="/tmp/seq",
    )
    assert result["success"] is False
    assert "reinforce" in result["error"]
    assert "sft" in result["error"] or "Must be one of" in result["error"]


@pytest.mark.asyncio
async def test_missing_dataset_path_returns_error():
    """Stage without dataset_path should fail validation."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[{"technique": "sft"}],
        output_dir="/tmp/seq",
    )
    assert result["success"] is False
    assert "dataset_path" in result["error"]


# ----------------------------------------------------------------
# Single stage tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch("finetuning_pipeline.services.training_service.TrainingService.train_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_single_sft_stage_success(mock_load, mock_train, tmp_path):
    """A single SFT stage should work end-to-end."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    out = str(tmp_path / "stage_1_sft")
    mock_train.return_value = _mock_train_result(out)

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[{"technique": "sft", "dataset_path": "/data.jsonl"}],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    assert result["success"] is True
    assert result["num_stages"] == 1
    assert len(result["stage_results"]) == 1
    assert result["stage_results"][0]["technique"] == "sft"
    assert "final_model_path" in result
    mock_train.assert_called_once()


# ----------------------------------------------------------------
# Multi-stage chaining tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch("finetuning_pipeline.services.sequential_service.SequentialTrainingService._merge_lora")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_dpo_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_sft_then_dpo_chains_model_path(
    mock_load, mock_train_sft, mock_train_dpo, mock_merge, tmp_path
):
    """Stage 1 (SFT) output should become stage 2 (DPO) base_model."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    sft_out = str(tmp_path / "stage_1_sft")
    merged_out = str(tmp_path / "stage_1_sft" / "merged")
    dpo_out = str(tmp_path / "stage_2_dpo")

    mock_train_sft.return_value = _mock_train_result(sft_out)
    mock_merge.return_value = merged_out
    mock_train_dpo.return_value = _mock_train_result(dpo_out)

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[
            {"technique": "sft", "dataset_path": "/sft.jsonl"},
            {"technique": "dpo", "dataset_path": "/dpo.jsonl", "beta": 0.2},
        ],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    assert result["success"] is True
    assert result["num_stages"] == 2

    # DPO should have received the merged SFT output as base_model
    dpo_call_kwargs = mock_train_dpo.call_args
    assert dpo_call_kwargs.kwargs["base_model"] == merged_out

    # LoRA merge should be called once (after SFT, not after DPO)
    assert mock_merge.call_count == 1


@pytest.mark.asyncio
@patch("finetuning_pipeline.services.sequential_service.SequentialTrainingService._merge_lora")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_grpo_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_dpo_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_three_stage_sft_dpo_grpo(
    mock_load, mock_train_sft, mock_train_dpo, mock_train_grpo, mock_merge, tmp_path
):
    """3-stage SFT→DPO→GRPO: merge after stages 1 and 2, not after 3."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    mock_train_sft.return_value = _mock_train_result(str(tmp_path / "s1"))
    mock_train_dpo.return_value = _mock_train_result(str(tmp_path / "s2"))
    mock_train_grpo.return_value = _mock_train_result(str(tmp_path / "s3"))
    mock_merge.return_value = str(tmp_path / "merged")

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[
            {"technique": "sft", "dataset_path": "/sft.jsonl"},
            {"technique": "dpo", "dataset_path": "/dpo.jsonl"},
            {"technique": "grpo", "dataset_path": "/grpo.jsonl"},
        ],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    assert result["success"] is True
    assert result["num_stages"] == 3
    assert mock_train_sft.call_count == 1
    assert mock_train_dpo.call_count == 1
    assert mock_train_grpo.call_count == 1
    # Merge after SFT (stage 1) and after DPO (stage 2), not after GRPO (last stage)
    assert mock_merge.call_count == 2


# ----------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch("finetuning_pipeline.services.sequential_service.SequentialTrainingService._merge_lora")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_dpo_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_stage_failure_stops_pipeline(
    mock_load, mock_train_sft, mock_train_dpo, mock_merge, tmp_path
):
    """If stage 1 fails, stage 2 should NOT run."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    mock_train_sft.return_value = _mock_train_failure("OOM")

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[
            {"technique": "sft", "dataset_path": "/sft.jsonl"},
            {"technique": "dpo", "dataset_path": "/dpo.jsonl"},
        ],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    assert result["success"] is False
    assert "Stage 1" in result["error"]
    assert len(result["stage_results"]) == 1
    mock_train_dpo.assert_not_called()
    mock_merge.assert_not_called()


@pytest.mark.asyncio
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_dataset_load_failure_stops_pipeline(mock_load, tmp_path):
    """If dataset loading fails, the stage should fail gracefully."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = {"success": False, "error": "File not found"}

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[{"technique": "sft", "dataset_path": "/missing.jsonl"}],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    assert result["success"] is False
    assert "failed to load" in result["error"].lower()


# ----------------------------------------------------------------
# Technique-specific parameter tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch("finetuning_pipeline.services.training_service.TrainingService.train_dpo_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_dpo_receives_beta_param(mock_load, mock_train_dpo, tmp_path):
    """DPO stage should pass beta parameter to train_dpo_model."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    mock_train_dpo.return_value = _mock_train_result(str(tmp_path / "s1"))

    svc = SequentialTrainingService()
    await svc.train_sequential(
        stages=[{"technique": "dpo", "dataset_path": "/d.jsonl", "beta": 0.3}],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    call_kwargs = mock_train_dpo.call_args.kwargs
    assert call_kwargs["beta"] == 0.3


@pytest.mark.asyncio
@patch("finetuning_pipeline.services.training_service.TrainingService.train_grpo_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_grpo_receives_num_generations_param(mock_load, mock_train_grpo, tmp_path):
    """GRPO stage should pass num_generations and omit LoRA params."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    mock_train_grpo.return_value = _mock_train_result(str(tmp_path / "s1"))

    svc = SequentialTrainingService()
    await svc.train_sequential(
        stages=[{"technique": "grpo", "dataset_path": "/g.jsonl", "num_generations": 8}],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    call_kwargs = mock_train_grpo.call_args.kwargs
    assert call_kwargs["num_generations"] == 8
    # GRPO should not receive LoRA params
    assert "use_lora" not in call_kwargs
    assert "lora_r" not in call_kwargs


# ----------------------------------------------------------------
# Merge control tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch("finetuning_pipeline.services.training_service.TrainingService.train_dpo_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_no_merge_when_flag_is_false(mock_load, mock_train_sft, mock_train_dpo, tmp_path):
    """merge_between_stages=False should skip LoRA merging."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    sft_out = str(tmp_path / "stage_1_sft")
    mock_train_sft.return_value = _mock_train_result(sft_out)
    mock_train_dpo.return_value = _mock_train_result(str(tmp_path / "stage_2_dpo"))

    svc = SequentialTrainingService()

    with patch.object(svc, "_merge_lora", new_callable=AsyncMock) as mock_merge:
        result = await svc.train_sequential(
            stages=[
                {"technique": "sft", "dataset_path": "/sft.jsonl"},
                {"technique": "dpo", "dataset_path": "/dpo.jsonl"},
            ],
            output_dir=str(tmp_path),
            base_model="fake-base",
            merge_between_stages=False,
        )

    assert result["success"] is True
    mock_merge.assert_not_called()
    # Without merge, DPO should get the raw SFT output_dir as base_model
    dpo_kwargs = mock_train_dpo.call_args.kwargs
    assert dpo_kwargs["base_model"] == sft_out


# ----------------------------------------------------------------
# Output structure tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch("finetuning_pipeline.services.training_service.TrainingService.train_model")
@patch("finetuning_pipeline.services.training_service.TrainingService.load_dataset_from_file")
async def test_output_structure(mock_load, mock_train, tmp_path):
    """Result dict should contain all expected keys."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    mock_load.return_value = _mock_load_result()
    mock_train.return_value = _mock_train_result(str(tmp_path / "s1"))

    svc = SequentialTrainingService()
    result = await svc.train_sequential(
        stages=[{"technique": "sft", "dataset_path": "/d.jsonl"}],
        output_dir=str(tmp_path),
        base_model="fake-base",
    )

    assert "success" in result
    assert "final_model_path" in result
    assert "base_model" in result
    assert "num_stages" in result
    assert "stage_results" in result
    assert "total_training_seconds" in result
    assert result["base_model"] == "fake-base"

    # Each stage result should have expected keys
    sr = result["stage_results"][0]
    assert "stage" in sr
    assert "technique" in sr
    assert "dataset_path" in sr
    assert "output_dir" in sr
    assert "training_result" in sr


@pytest.mark.asyncio
async def test_sequential_merge_lora_avoids_auto_device_map_and_falls_back(tmp_path):
    """Merge should avoid device_map='auto' and retry on CPU if single-device load fails."""
    from finetuning_pipeline.services.sequential_service import SequentialTrainingService

    stage_dir = tmp_path / "stage_1_sft"
    stage_dir.mkdir()

    base_model = MagicMock()
    peft_model = MagicMock()
    merged_model = MagicMock()
    tokenizer = MagicMock()
    peft_model.merge_and_unload.return_value = merged_model

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.empty_cache"
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=[RuntimeError("oom"), base_model],
    ) as mock_load_model, patch(
        "peft.PeftModel.from_pretrained", return_value=peft_model
    ), patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ):
        result = await SequentialTrainingService._merge_lora(
            str(stage_dir), "fake-base", "fake-tokenizer"
        )

    assert result == str(stage_dir / "merged")
    assert mock_load_model.call_count == 2
    assert mock_load_model.call_args_list[0].kwargs["device_map"] == {"": 0}
    assert "device_map" not in mock_load_model.call_args_list[1].kwargs
