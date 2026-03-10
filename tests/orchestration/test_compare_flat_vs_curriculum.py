"""Tests for compare_flat_vs_curriculum workflow (Feature 3)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from orchestration.workflow import PipelineOrchestrator


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _make_orchestrator(**overrides) -> PipelineOrchestrator:
    """Build PipelineOrchestrator with all AsyncMock services."""
    defaults = dict(
        generator=AsyncMock(),
        cleaner=AsyncMock(),
        normalizer=AsyncMock(),
        evaluator=AsyncMock(),
        finetuner=AsyncMock(),
        hoster=AsyncMock(),
    )
    defaults.update(overrides)
    return PipelineOrchestrator(**defaults)


def _mock_load_success(dataset_obj=None):
    return {
        "success": True,
        "dataset_object": dataset_obj or [{"instruction": "q", "output": "a"}],
        "num_examples": 1,
    }


def _mock_train_success(model_path: str):
    return {"success": True, "model_path": model_path, "num_training_examples": 3}


def _mock_curriculum_success(final_path: str):
    return {
        "success": True,
        "final_model_path": final_path,
        "num_stages": 3,
        "stage_results": [],
    }


def _mock_orchestration_dataset():
    return [{"instruction": "Use tools", "output": "Step 1: search()"}]


# ----------------------------------------------------------------
# Core workflow tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_returns_both_training_results():
    """Both flat SFT and curriculum results should be in the output."""
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value=_mock_load_success())
    finetuner.train_model = AsyncMock(
        return_value=_mock_train_success("/out/flat_sft")
    )
    finetuner.train_curriculum_model = AsyncMock(
        return_value=_mock_curriculum_success("/out/curriculum_sft/stage_3")
    )

    orch = _make_orchestrator(finetuner=finetuner)
    result = await orch.compare_flat_vs_curriculum(
        dataset_path="/data.jsonl",
        output_dir="/out",
    )

    assert result["success"] is True
    assert result["flat_sft"]["model_path"] == "/out/flat_sft"
    assert result["curriculum_sft"]["model_path"] == "/out/curriculum_sft/stage_3"
    finetuner.train_model.assert_called_once()
    finetuner.train_curriculum_model.assert_called_once()


@pytest.mark.asyncio
async def test_compare_with_test_prompts_runs_comparison():
    """When test_prompts is provided, compare_models should be called."""
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value=_mock_load_success())
    finetuner.train_model = AsyncMock(
        return_value=_mock_train_success("/out/flat")
    )
    finetuner.train_curriculum_model = AsyncMock(
        return_value=_mock_curriculum_success("/out/curriculum/stage_3")
    )
    finetuner.compare_models = AsyncMock(
        return_value={"success": True, "comparisons": []}
    )

    orch = _make_orchestrator(finetuner=finetuner)
    result = await orch.compare_flat_vs_curriculum(
        dataset_path="/data.jsonl",
        output_dir="/out",
        test_prompts=["Hello", "Explain AI"],
    )

    assert result["comparison"] is not None
    assert "flat_vs_base" in result["comparison"]
    assert "curriculum_vs_base" in result["comparison"]
    # compare_models called twice: flat vs base, curriculum vs base
    assert finetuner.compare_models.call_count == 2


@pytest.mark.asyncio
async def test_compare_without_test_data_skips_comparison():
    """No test_prompts and no test_data_path → comparison section is None."""
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value=_mock_load_success())
    finetuner.train_model = AsyncMock(
        return_value=_mock_train_success("/out/flat")
    )
    finetuner.train_curriculum_model = AsyncMock(
        return_value=_mock_curriculum_success("/out/curriculum/stage_3")
    )

    orch = _make_orchestrator(finetuner=finetuner)
    result = await orch.compare_flat_vs_curriculum(
        dataset_path="/data.jsonl",
        output_dir="/out",
    )

    assert result["comparison"] is None
    finetuner.compare_models.assert_not_called()


@pytest.mark.asyncio
async def test_compare_returns_error_on_dataset_load_failure():
    """Failed dataset load should return error immediately."""
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(
        return_value={"success": False, "error": "File not found"}
    )

    orch = _make_orchestrator(finetuner=finetuner)
    result = await orch.compare_flat_vs_curriculum(
        dataset_path="/missing.jsonl",
        output_dir="/out",
    )

    assert result["success"] is False
    assert "load" in result["error"].lower()


@pytest.mark.asyncio
async def test_compare_still_returns_when_one_training_fails():
    """If flat SFT fails but curriculum succeeds, result should contain both."""
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value=_mock_load_success())
    finetuner.train_model = AsyncMock(
        return_value={"success": False, "error": "OOM"}
    )
    finetuner.train_curriculum_model = AsyncMock(
        return_value=_mock_curriculum_success("/out/curriculum/stage_3")
    )

    orch = _make_orchestrator(finetuner=finetuner)
    result = await orch.compare_flat_vs_curriculum(
        dataset_path="/data.jsonl",
        output_dir="/out",
    )

    # Still returns success (partial results are valid)
    assert result["success"] is True
    assert result["flat_sft"]["model_path"] is None  # flat failed
    assert result["curriculum_sft"]["model_path"] == "/out/curriculum/stage_3"


@pytest.mark.asyncio
async def test_compare_output_structure():
    """Result dict should contain all expected top-level keys."""
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value=_mock_load_success())
    finetuner.train_model = AsyncMock(
        return_value=_mock_train_success("/out/flat")
    )
    finetuner.train_curriculum_model = AsyncMock(
        return_value=_mock_curriculum_success("/out/curriculum/stage_3")
    )

    orch = _make_orchestrator(finetuner=finetuner)
    result = await orch.compare_flat_vs_curriculum(
        dataset_path="/data.jsonl",
        output_dir="/out",
        base_model="my-base",
        num_stages=3,
        num_epochs_flat=2,
    )

    expected_keys = {
        "success", "dataset_path", "base_model",
        "flat_sft", "curriculum_sft", "comparison",
    }
    assert expected_keys.issubset(result.keys())
    assert result["base_model"] == "my-base"
    assert result["flat_sft"]["total_epochs"] == 2
    assert result["curriculum_sft"]["num_stages"] == 3


@pytest.mark.asyncio
async def test_train_orchestrator_uses_provided_training_data_for_sft():
    finetuner = AsyncMock()
    finetuner.train_model = AsyncMock(return_value=_mock_train_success("/out/model"))

    orch = _make_orchestrator(
        finetuner=finetuner,
        orchestration_data_service=AsyncMock(),
    )
    orch.orchestration_data_service.export = AsyncMock(return_value={"success": True})

    result = await orch.train_orchestrator(
        domain_description="tool routing",
        agent=AsyncMock(),
        output_dir="/out",
        output_format="sft",
        training_data=_mock_orchestration_dataset(),
    )

    assert result["success"] is True
    assert result["used_provided_training_data"] is True
    finetuner.train_model.assert_awaited_once()
    orch.orchestration_data_service.generate_problems.assert_not_called()
    orch.orchestration_data_service.collect_trajectories.assert_not_called()
    orch.orchestration_data_service.build_training_data.assert_not_called()


@pytest.mark.asyncio
async def test_train_orchestrator_uses_dpo_trainer_for_dpo_format():
    finetuner = AsyncMock()
    finetuner.train_dpo_model = AsyncMock(return_value=_mock_train_success("/out/model"))

    orch = _make_orchestrator(
        finetuner=finetuner,
        orchestration_data_service=AsyncMock(),
    )
    orch.orchestration_data_service.export = AsyncMock(return_value={"success": True})

    result = await orch.train_orchestrator(
        domain_description="tool routing",
        agent=AsyncMock(),
        output_dir="/out",
        output_format="dpo",
        training_data=[{"prompt": "Task", "chosen": "Best", "rejected": "Worst"}],
    )

    assert result["success"] is True
    finetuner.train_dpo_model.assert_awaited_once()
    finetuner.train_model.assert_not_called()


@pytest.mark.asyncio
async def test_train_orchestrator_uses_grpo_trainer_for_grpo_format():
    finetuner = AsyncMock()
    finetuner.train_grpo_model = AsyncMock(return_value=_mock_train_success("/out/model"))

    orch = _make_orchestrator(
        finetuner=finetuner,
        orchestration_data_service=AsyncMock(),
    )
    orch.orchestration_data_service.export = AsyncMock(return_value={"success": True})

    result = await orch.train_orchestrator(
        domain_description="tool routing",
        agent=AsyncMock(),
        output_dir="/out",
        output_format="grpo",
        training_data=[{"prompt": "Task", "responses": ["A"], "rewards": [1.0]}],
    )

    assert result["success"] is True
    finetuner.train_grpo_model.assert_awaited_once()
    finetuner.train_model.assert_not_called()
