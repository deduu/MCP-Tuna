from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock

from orchestration.workflow import PipelineOrchestrator


def _make_orchestrator(**overrides) -> PipelineOrchestrator:
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


@pytest.mark.asyncio
async def test_generate_and_evaluate_accepts_multiple_files():
    generator = AsyncMock()
    generator.generate_from_document = AsyncMock(
        side_effect=[
            {"success": True, "data_points": [{"instruction": "q1", "output": "a1"}]},
            {"success": True, "data_points": [{"instruction": "q2", "output": "a2"}]},
        ]
    )
    cleaner = AsyncMock()
    cleaner.clean_dataset = AsyncMock(
        return_value={
            "success": True,
            "data_points": [
                {"instruction": "q1", "output": "a1"},
                {"instruction": "q2", "output": "a2"},
            ],
            "cleaned_count": 2,
        }
    )
    normalizer = AsyncMock()
    normalizer.normalize_dataset = AsyncMock(
        return_value={
            "success": True,
            "data_points": [
                {"instruction": "q1", "output": "a1"},
                {"instruction": "q2", "output": "a2"},
            ],
            "count": 2,
        }
    )
    evaluator = AsyncMock()
    evaluator.evaluate_dataset = AsyncMock(
        return_value={
            "success": True,
            "data_points": [
                {"instruction": "q1", "output": "a1", "weighted_score": 0.9},
                {"instruction": "q2", "output": "a2", "weighted_score": 0.8},
            ],
            "count": 2,
        }
    )
    evaluator.filter_by_quality = AsyncMock(
        return_value={
            "success": True,
            "data_points": [
                {"instruction": "q1", "output": "a1", "weighted_score": 0.9},
                {"instruction": "q2", "output": "a2", "weighted_score": 0.8},
            ],
            "filtered_count": 2,
        }
    )
    evaluator.analyze_statistics = AsyncMock(return_value={"success": True, "statistics": {"weighted": {"mean": 0.85}}})

    orch = _make_orchestrator(
        generator=generator,
        cleaner=cleaner,
        normalizer=normalizer,
        evaluator=evaluator,
    )
    result = await orch.generate_and_evaluate(file_paths=["/docs/a.md", "/docs/b.md"])

    assert result["success"] is True
    assert result["file_paths"] == ["/docs/a.md", "/docs/b.md"]
    assert result["per_file_generated"] == {"/docs/a.md": 1, "/docs/b.md": 1}
    assert result["pipeline_stages"]["generated"] == 2
    assert generator.generate_from_document.await_count == 2


@pytest.mark.asyncio
async def test_full_pipeline_reports_progress_for_multi_doc_run():
    progress_updates: list[tuple[str, int, float]] = []

    async def on_progress(**kwargs):
        progress_updates.append(
            (
                str(kwargs.get("current_stage")),
                int(kwargs.get("current_step", 0)),
                float(kwargs.get("percent_complete", 0)),
            )
        )

    generator = AsyncMock()
    generator.generate_from_document = AsyncMock(
        side_effect=[
            {"success": True, "data_points": [{"instruction": "q1", "output": "a1"}]},
            {"success": True, "data_points": [{"instruction": "q2", "output": "a2"}]},
        ]
    )
    generator.export_dataset = AsyncMock(return_value={"success": True, "file_path": "/out/dataset.jsonl"})
    cleaner = AsyncMock()
    cleaner.clean_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "cleaned_count": 1}
    )
    normalizer = AsyncMock()
    normalizer.normalize_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "count": 1}
    )
    evaluator = AsyncMock()
    evaluator.evaluate_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "count": 1}
    )
    evaluator.filter_by_quality = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "filtered_count": 1}
    )
    evaluator.analyze_statistics = AsyncMock(return_value={"success": True, "statistics": {}})
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value={"success": True, "dataset_object": [{"prompt": "q1", "response": "a1"}]})
    finetuner.train_model = AsyncMock(return_value={"success": True, "model_path": "/out/model"})
    finetuner.run_inference = AsyncMock(return_value={"success": True, "outputs": ["ok"]})

    orch = _make_orchestrator(
        generator=generator,
        cleaner=cleaner,
        normalizer=normalizer,
        evaluator=evaluator,
        finetuner=finetuner,
    )
    result = await orch.full_pipeline(
        file_paths=["/docs/a.md", "/docs/b.md"],
        output_dir="/out",
        progress_callback=on_progress,
    )

    assert result["success"] is True
    assert result["file_paths"] == ["/docs/a.md", "/docs/b.md"]
    stage_names = [stage for stage, _, _ in progress_updates]
    assert stage_names[:3] == ["generate", "generate", "generate"]
    assert stage_names[3:] == [
        "clean", "normalize", "evaluate", "filter", "export", "train", "test", "deploy",
    ]


@pytest.mark.asyncio
async def test_generate_stage_emits_heartbeat_updates_for_long_running_work():
    messages: list[str] = []

    async def on_progress(**kwargs):
        if kwargs.get("current_stage") == "generate" and kwargs.get("status_message"):
            messages.append(str(kwargs["status_message"]))

    async def slow_generate(**kwargs):
        await asyncio.sleep(2.2)
        return {"success": True, "data_points": [{"instruction": "q1", "output": "a1"}]}

    generator = AsyncMock()
    generator.generate_from_document = AsyncMock(side_effect=slow_generate)
    cleaner = AsyncMock()
    cleaner.clean_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "cleaned_count": 1}
    )
    normalizer = AsyncMock()
    normalizer.normalize_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "count": 1}
    )
    evaluator = AsyncMock()
    evaluator.evaluate_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "count": 1}
    )
    evaluator.filter_by_quality = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "filtered_count": 1}
    )
    evaluator.analyze_statistics = AsyncMock(return_value={"success": True, "statistics": {}})

    orch = _make_orchestrator(
        generator=generator,
        cleaner=cleaner,
        normalizer=normalizer,
        evaluator=evaluator,
    )
    result = await orch.generate_and_evaluate(
        file_path="/docs/slow.jsonl",
        progress_callback=on_progress,
    )

    assert result["success"] is True
    assert any("Generating from file 1 of 1" in message for message in messages)
    assert any("elapsed" in message for message in messages)


@pytest.mark.asyncio
async def test_generate_and_evaluate_surfaces_page_failure_when_no_rows_are_generated():
    generator = AsyncMock()
    generator.generate_from_document = AsyncMock(
        return_value={
            "success": True,
            "data_points": [],
            "page_errors": [
                {"page": 1, "page_index": 0, "error": "Invalid JSON returned: not-json"},
            ],
        }
    )

    orch = _make_orchestrator(generator=generator)
    result = await orch.generate_and_evaluate(file_path="/docs/a.md")

    assert result["success"] is False
    assert result["error"] == (
        "No data points generated from provided files. "
        "First page failure in a.md page 1: Invalid JSON returned: not-json"
    )
    assert result["generation_errors"] == [
        {
            "file_path": "/docs/a.md",
            "file_name": "a.md",
            "page": 1,
            "page_index": 0,
            "error": "Invalid JSON returned: not-json",
        }
    ]


@pytest.mark.asyncio
async def test_full_pipeline_returns_failure_when_training_fails():
    generator = AsyncMock()
    generator.generate_from_document = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}]}
    )
    generator.export_dataset = AsyncMock(return_value={"success": True})
    cleaner = AsyncMock()
    cleaner.clean_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "cleaned_count": 1}
    )
    normalizer = AsyncMock()
    normalizer.normalize_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "count": 1}
    )
    evaluator = AsyncMock()
    evaluator.evaluate_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "count": 1}
    )
    evaluator.filter_by_quality = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "filtered_count": 1}
    )
    evaluator.analyze_statistics = AsyncMock(return_value={"success": True, "statistics": {}})
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value={"success": True, "dataset_object": [{"prompt": "q1", "response": "a1"}]})
    finetuner.train_model = AsyncMock(return_value={"success": False, "error": "OOM"})

    orch = _make_orchestrator(
        generator=generator,
        cleaner=cleaner,
        normalizer=normalizer,
        evaluator=evaluator,
        finetuner=finetuner,
    )
    result = await orch.full_pipeline(file_path="/docs/a.md", output_dir="/out")

    assert result["success"] is False
    assert result["error"] == "OOM"
    assert result["training"]["success"] is False


@pytest.mark.asyncio
async def test_full_pipeline_deploys_direct_model_when_lora_disabled():
    generator = AsyncMock()
    generator.generate_from_document = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}]}
    )
    generator.export_dataset = AsyncMock(return_value={"success": True})
    cleaner = AsyncMock()
    cleaner.clean_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "cleaned_count": 1}
    )
    normalizer = AsyncMock()
    normalizer.normalize_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "count": 1}
    )
    evaluator = AsyncMock()
    evaluator.evaluate_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "count": 1}
    )
    evaluator.filter_by_quality = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "filtered_count": 1}
    )
    evaluator.analyze_statistics = AsyncMock(return_value={"success": True, "statistics": {}})
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value={"success": True, "dataset_object": [{"prompt": "q1", "response": "a1"}]})
    finetuner.train_model = AsyncMock(
        return_value={"success": True, "model_path": "/out/model", "config": {"trainer": "sft", "use_lora": False}}
    )
    finetuner.run_inference = AsyncMock(return_value={"success": True, "outputs": ["ok"]})
    hoster = AsyncMock()
    hoster.deploy_as_mcp = AsyncMock(return_value={"success": True, "endpoint": "http://127.0.0.1:8001"})

    orch = _make_orchestrator(
        generator=generator,
        cleaner=cleaner,
        normalizer=normalizer,
        evaluator=evaluator,
        finetuner=finetuner,
        hoster=hoster,
    )
    result = await orch.full_pipeline(file_path="/docs/a.md", output_dir="/out", use_lora=False, deploy=True)

    assert result["success"] is True
    finetuner.run_inference.assert_awaited_once_with(
        prompts=["q1"],
        model_path="/out/model",
        adapter_path=None,
    )
    config = hoster.deploy_as_mcp.await_args.args[0]
    assert config.model_path == "/out/model"
    assert config.adapter_path is None


@pytest.mark.asyncio
async def test_full_pipeline_passes_push_to_hub_to_training():
    generator = AsyncMock()
    generator.generate_from_document = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}]}
    )
    generator.export_dataset = AsyncMock(return_value={"success": True})
    cleaner = AsyncMock()
    cleaner.clean_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "cleaned_count": 1}
    )
    normalizer = AsyncMock()
    normalizer.normalize_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1"}], "count": 1}
    )
    evaluator = AsyncMock()
    evaluator.evaluate_dataset = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "count": 1}
    )
    evaluator.filter_by_quality = AsyncMock(
        return_value={"success": True, "data_points": [{"instruction": "q1", "output": "a1", "weighted_score": 0.9}], "filtered_count": 1}
    )
    evaluator.analyze_statistics = AsyncMock(return_value={"success": True, "statistics": {}})
    finetuner = AsyncMock()
    finetuner.load_dataset_from_file = AsyncMock(return_value={"success": True, "dataset_object": [{"prompt": "q1", "response": "a1"}]})
    finetuner.train_model = AsyncMock(return_value={"success": True, "model_path": "/out/model"})
    finetuner.run_inference = AsyncMock(return_value={"success": True, "outputs": ["ok"]})

    orch = _make_orchestrator(
        generator=generator,
        cleaner=cleaner,
        normalizer=normalizer,
        evaluator=evaluator,
        finetuner=finetuner,
    )
    result = await orch.full_pipeline(
        file_path="/docs/a.md",
        output_dir="/out",
        push_to_hub="my-org/my-model",
    )

    assert result["success"] is True
    train_kwargs = finetuner.train_model.await_args.kwargs
    assert train_kwargs["push_to_hub"] == "my-org/my-model"
