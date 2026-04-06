from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
async def test_benchmark_finetuning_runs_multi_pack_comparison_and_saves_report(tmp_path):
    datasets = {
        "/train.jsonl": [{"prompt": "train", "response": "ok"}],
        "/stage1.jsonl": [{"prompt": "easy", "response": "ok"}],
        "/stage2.jsonl": [{"prompt": "medium", "response": "ok"}],
        "/dev.jsonl": [
            {
                "system": "Be concise.",
                "user": "What is Salestify?",
                "assistant": "Salestify helps manage WhatsApp sales chats.",
                "required_terms": ["whatsapp", "sales"],
            }
        ],
        "/holdout.jsonl": [
            {
                "prompt": "Owner sulit memantau follow-up tim.",
                "response": "Salestify helps owner monitor follow-up and chat visibility.",
                "required_terms": ["owner", "visibility"],
            }
        ],
        "/safety.jsonl": [
            {
                "prompt": "Berapa harga paketnya?",
                "response": "Silakan hubungi tim sales untuk info harga.",
                "required_terms": ["sales"],
                "forbidden_terms": ["Rp", "$"],
            }
        ],
    }

    finetuner = AsyncMock()

    async def load_side_effect(file_path: str, format: str = "json"):
        rows = datasets[file_path]
        return {
            "success": True,
            "dataset_object": rows,
            "num_examples": len(rows),
            "format": format,
        }

    async def run_inference_side_effect(
        prompts,
        model_path: str,
        adapter_path=None,
        **kwargs,
    ):
        candidate = str(adapter_path or model_path)
        outputs = {
            "/models/flat": {
                "What is Salestify?": "Salestify helps sales teams.",
                "Owner sulit memantau follow-up tim.": "Salestify gives visibility for owner follow-up.",
                "Berapa harga paketnya?": "Hubungi sales untuk info harga.",
            },
            "/models/curriculum": {
                "What is Salestify?": "Salestify helps manage WhatsApp sales chats.",
                "Owner sulit memantau follow-up tim.": "Salestify helps owner monitor follow-up and chat visibility.",
                "Berapa harga paketnya?": "Silakan hubungi tim sales untuk info harga.",
            },
            "/models/notebook": {
                "What is Salestify?": "Salestify is a WhatsApp sales assistant.",
                "Owner sulit memantau follow-up tim.": "Salestify gives management visibility.",
                "Berapa harga paketnya?": "Silakan hubungi sales untuk info harga.",
            },
        }
        rows = []
        for prompt in prompts:
            rows.append(
                {
                    "prompt": prompt,
                    "response": outputs[candidate][prompt],
                    "generation_time_seconds": 0.1,
                    "tokens_generated": 8,
                    "tokens_per_second": 80.0,
                }
            )
        return {"success": True, "results": rows}

    finetuner.load_dataset_from_file = AsyncMock(side_effect=load_side_effect)
    finetuner.train_model = AsyncMock(
        return_value={
            "success": True,
            "model_path": "/models/flat",
            "config": {"use_lora": True},
        }
    )
    finetuner.train_curriculum_model = AsyncMock(
        return_value={
            "success": True,
            "final_model_path": "/models/curriculum",
            "adapter_base_model": "base-model",
            "stage_results": [
                {
                    "training_result": {
                        "success": True,
                        "config": {"use_lora": True},
                    }
                }
            ],
        }
    )
    finetuner.run_inference = AsyncMock(side_effect=run_inference_side_effect)

    orch = _make_orchestrator(finetuner=finetuner)
    output_dir = tmp_path / "benchmark"
    result = await orch.benchmark_finetuning(
        train_dataset_path="/train.jsonl",
        output_dir=str(output_dir),
        base_model="base-model",
        stage_dataset_paths=["/stage1.jsonl", "/stage2.jsonl"],
        eval_file_path="/dev.jsonl",
        dev_data_path="/dev.jsonl",
        holdout_data_path="/holdout.jsonl",
        safety_data_path="/safety.jsonl",
        seeds=[7],
        lora_dropout=0.15,
        weight_decay=0.02,
        save_best_model=False,
        reference_models=[
            {
                "name": "notebook_flat",
                "model_path": "base-model",
                "adapter_path": "/models/notebook",
            }
        ],
        benchmark_gates=[
            {
                "name": "curriculum_beats_flat",
                "candidate_method": "curriculum_sft",
                "baseline_method": "flat_sft",
                "pack": "hidden_holdout",
                "metric": "avg_composite_score",
                "min_delta": 0.05,
            }
        ],
    )

    assert result["success"] is True
    assert result["summary"]["best_method"] == "curriculum_sft"
    assert result["summary"]["best_run"]["method"] == "curriculum_sft"
    assert result["summary"]["best_run"]["seed"] == 7
    assert result["summary"]["gates"][0]["passed"] is True
    assert result["evaluation_packs"]["hidden_holdout"]["num_cases"] == 1
    assert Path(result["results_path"]).exists()

    saved = json.loads(Path(result["results_path"]).read_text(encoding="utf-8"))
    assert saved["summary"]["best_method"] == "curriculum_sft"
    assert saved["benchmark_config"]["eval_do_sample"] is False

    flat_kwargs = finetuner.train_model.await_args.kwargs
    assert flat_kwargs["eval_file_path"] == "/dev.jsonl"
    assert flat_kwargs["lora_dropout"] == 0.15
    assert flat_kwargs["weight_decay"] == 0.02
    assert flat_kwargs["save_best_model"] is False
    assert flat_kwargs["dataset_path"] == "/train.jsonl"
    assert flat_kwargs["run_source"] == "workflow.benchmark_finetuning"

    curriculum_kwargs = finetuner.train_curriculum_model.await_args.kwargs
    assert curriculum_kwargs["stage_datasets"] == [
        datasets["/stage1.jsonl"],
        datasets["/stage2.jsonl"],
    ]
    assert curriculum_kwargs["eval_file_path"] == "/dev.jsonl"
    assert curriculum_kwargs["lora_dropout"] == 0.15
    assert curriculum_kwargs["weight_decay"] == 0.02
    assert curriculum_kwargs["save_best_model"] is False
    assert curriculum_kwargs["dataset_path"] == "/train.jsonl"
    assert curriculum_kwargs["run_source"] == "workflow.benchmark_finetuning"
    assert finetuner.run_inference.await_count == 9
    assert all(
        call.kwargs["do_sample"] is False
        for call in finetuner.run_inference.await_args_list
    )


@pytest.mark.asyncio
async def test_benchmark_finetuning_requires_eval_pack(tmp_path):
    orch = _make_orchestrator(finetuner=AsyncMock())

    result = await orch.benchmark_finetuning(
        train_dataset_path="/train.jsonl",
        output_dir=str(tmp_path / "out"),
    )

    assert result["success"] is False
    assert "evaluation pack" in result["error"].lower()


@pytest.mark.asyncio
async def test_benchmark_finetuning_supports_dpo_and_grpo_candidates(tmp_path):
    datasets = {
        "/dpo.jsonl": [
            {
                "prompt": "Apa itu Salestify?",
                "chosen": "Salestify membantu tim sales mengelola follow-up WhatsApp.",
                "rejected": "Saya tidak tahu.",
            }
        ],
        "/grpo.jsonl": [
            {
                "prompt": "Apa itu Salestify?",
                "responses": [
                    "Salestify membantu tim sales mengelola follow-up WhatsApp.",
                    "Salestify dipakai untuk mengatur chat sales.",
                ],
                "rewards": [1.0, 0.8],
            }
        ],
        "/dev.jsonl": [
            {
                "prompt": "Apa itu Salestify?",
                "response": "Salestify membantu tim sales mengelola follow-up WhatsApp.",
                "required_terms": ["Salestify", "sales"],
            }
        ],
    }

    finetuner = AsyncMock()

    async def load_side_effect(file_path: str, format: str = "json"):
        rows = datasets[file_path]
        return {
            "success": True,
            "dataset_object": rows,
            "num_examples": len(rows),
            "format": format,
        }

    async def run_inference_side_effect(
        prompts,
        model_path: str,
        adapter_path=None,
        **kwargs,
    ):
        candidate = str(adapter_path or model_path)
        outputs = {
            "/models/mcp_dpo": {
                "Apa itu Salestify?": "Salestify membantu tim sales mengelola follow-up WhatsApp.",
            },
            "/models/mcp_grpo": {
                "Apa itu Salestify?": "Salestify membantu tim sales memantau follow-up WhatsApp.",
            },
            "/models/notebook_dpo": {
                "Apa itu Salestify?": "Salestify membantu tim sales mengatur chat WhatsApp.",
            },
            "/models/notebook_grpo": {
                "Apa itu Salestify?": "Salestify memberi tim sales visibilitas follow-up.",
            },
        }
        return {
            "success": True,
            "results": [
                {
                    "prompt": prompt,
                    "response": outputs[candidate][prompt],
                    "generation_time_seconds": 0.1,
                    "tokens_generated": 8,
                    "tokens_per_second": 80.0,
                }
                for prompt in prompts
            ],
        }

    finetuner.load_dataset_from_file = AsyncMock(side_effect=load_side_effect)
    finetuner.train_dpo_model = AsyncMock(
        return_value={
            "success": True,
            "model_path": "/models/mcp_dpo",
            "config": {"trainer": "dpo", "use_lora": True},
        }
    )
    finetuner.train_grpo_model = AsyncMock(
        return_value={
            "success": True,
            "model_path": "/models/mcp_grpo",
            "config": {"trainer": "grpo"},
        }
    )
    finetuner.run_inference = AsyncMock(side_effect=run_inference_side_effect)

    orch = _make_orchestrator(finetuner=finetuner)
    result = await orch.benchmark_finetuning(
        train_dataset_path="/unused-default.jsonl",
        output_dir=str(tmp_path / "benchmark"),
        base_model="base-model",
        dev_data_path="/dev.jsonl",
        training_methods=[
            {
                "name": "mcp_dpo",
                "method": "dpo",
                "dataset_path": "/dpo.jsonl",
                "adapter_path": "/models/best_sft",
                "num_epochs": 2,
                "beta": 0.2,
            },
            {
                "name": "mcp_grpo",
                "method": "grpo",
                "dataset_path": "/grpo.jsonl",
                "adapter_path": "/models/best_sft",
                "num_epochs": 1,
                "num_generations": 6,
            },
        ],
        reference_models=[
            {
                "name": "notebook_dpo",
                "model_path": "base-model",
                "adapter_path": "/models/notebook_dpo",
            },
            {
                "name": "notebook_grpo",
                "model_path": "/models/notebook_grpo",
            },
        ],
        seeds=[13],
    )

    assert result["success"] is True
    assert {run["method"] for run in result["runs"]} == {
        "mcp_dpo",
        "mcp_grpo",
        "notebook_dpo",
        "notebook_grpo",
    }
    assert result["methods"]["training_methods"] == [
        {
            "name": "mcp_dpo",
            "trainer": "dpo",
            "dataset_path": "/dpo.jsonl",
            "stage_dataset_paths": [],
        },
        {
            "name": "mcp_grpo",
            "trainer": "grpo",
            "dataset_path": "/grpo.jsonl",
            "stage_dataset_paths": [],
        },
    ]

    finetuner.train_dpo_model.assert_awaited_once()
    finetuner.train_grpo_model.assert_awaited_once()
    finetuner.train_model.assert_not_called()
    finetuner.train_curriculum_model.assert_not_called()

    dpo_kwargs = finetuner.train_dpo_model.await_args.kwargs
    assert dpo_kwargs["dataset"] == datasets["/dpo.jsonl"]
    assert dpo_kwargs["adapter_path"] == "/models/best_sft"
    assert dpo_kwargs["num_epochs"] == 2
    assert dpo_kwargs["beta"] == 0.2

    grpo_kwargs = finetuner.train_grpo_model.await_args.kwargs
    assert grpo_kwargs["dataset"] == datasets["/grpo.jsonl"]
    assert grpo_kwargs["adapter_path"] == "/models/best_sft"
    assert grpo_kwargs["num_epochs"] == 1
    assert grpo_kwargs["num_generations"] == 6
    assert all(
        call.kwargs["do_sample"] is False
        for call in finetuner.run_inference.await_args_list
    )


@pytest.mark.asyncio
async def test_benchmark_finetuning_can_isolate_eval_process(tmp_path):
    datasets = {
        "/dev.jsonl": [
            {
                "prompt": "Apa itu Salestify?",
                "response": "Salestify membantu tim sales mengelola follow-up WhatsApp.",
                "required_terms": ["Salestify", "sales"],
            }
        ],
    }

    finetuner = AsyncMock()

    async def load_side_effect(file_path: str, format: str = "json"):
        return {
            "success": True,
            "dataset_object": datasets[file_path],
            "num_examples": len(datasets[file_path]),
            "format": format,
        }

    finetuner.load_dataset_from_file = AsyncMock(side_effect=load_side_effect)
    finetuner.run_inference = AsyncMock()

    orch = _make_orchestrator(finetuner=finetuner)
    orch._evaluate_benchmark_candidate_in_subprocess = lambda **kwargs: {
        "success": True,
        "packs": {
            "dev": {
                "cases": [
                    {
                        "prompt": "Apa itu Salestify?",
                        "generated_response": "Salestify membantu tim sales mengelola follow-up WhatsApp.",
                        "reference_response": "Salestify membantu tim sales mengelola follow-up WhatsApp.",
                        "composite_score": 1.0,
                    }
                ],
                "summary": {
                    "num_cases": 1,
                    "avg_composite_score": 1.0,
                },
            }
        },
    }

    result = await orch.benchmark_finetuning(
        train_dataset_path="/unused.jsonl",
        output_dir=str(tmp_path / "benchmark"),
        base_model="base-model",
        dev_data_path="/dev.jsonl",
        training_methods=[],
        reference_models=[
            {
                "name": "notebook_dpo",
                "model_path": "base-model",
                "adapter_path": "/models/notebook_dpo",
            }
        ],
        eval_process_isolation=True,
    )

    assert result["success"] is True
    assert result["benchmark_config"]["eval_process_isolation"] is True
    finetuner.run_inference.assert_not_called()
