"""Integration tests for new MCP tools added to the gateway."""

from __future__ import annotations

import json
import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gateway():
    """Instantiate a gateway without loading environment side effects."""
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway
        return TunaGateway()


def _get_gateway_tool_names() -> set[str]:
    """Instantiate gateway and return all registered tool names."""
    gw = _make_gateway()
    return set(gw.mcp._tools.keys())


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Verify all new tools are registered on the gateway."""

    @pytest.fixture(scope="class")
    def tool_names(self) -> set[str]:
        return _get_gateway_tool_names()

    # Dataset tools
    def test_dataset_save_registered(self, tool_names):
        assert "dataset.save" in tool_names

    def test_dataset_load_registered(self, tool_names):
        assert "dataset.load" in tool_names

    def test_dataset_preview_registered(self, tool_names):
        assert "dataset.preview" in tool_names

    def test_dataset_info_registered(self, tool_names):
        assert "dataset.info" in tool_names

    def test_dataset_delete_registered(self, tool_names):
        assert "dataset.delete" in tool_names

    def test_dataset_split_registered(self, tool_names):
        assert "dataset.split" in tool_names

    def test_dataset_merge_registered(self, tool_names):
        assert "dataset.merge" in tool_names

    def test_file_upload_registered(self, tool_names):
        assert "file.upload" in tool_names

    # Generate tools
    def test_generate_from_text_registered(self, tool_names):
        assert "generate.from_text" in tool_names

    def test_generate_from_hf_dataset_registered(self, tool_names):
        assert "generate.from_hf_dataset" in tool_names

    def test_generate_list_profiles_registered(self, tool_names):
        assert "generate.list_profiles" in tool_names

    def test_generate_get_profile_registered(self, tool_names):
        assert "generate.get_profile" in tool_names

    def test_generate_preview_composition_registered(self, tool_names):
        assert "generate.preview_composition" in tool_names

    def test_generate_compose_profiled_dataset_registered(self, tool_names):
        assert "generate.compose_profiled_dataset" in tool_names

    def test_generate_list_schema_adapters_registered(self, tool_names):
        assert "generate.list_schema_adapters" in tool_names

    def test_generate_register_schema_adapter_registered(self, tool_names):
        assert "generate.register_schema_adapter" in tool_names

    def test_generate_list_hf_recipes_registered(self, tool_names):
        assert "generate.list_hf_recipes" in tool_names

    def test_generate_get_hf_recipe_registered(self, tool_names):
        assert "generate.get_hf_recipe" in tool_names

    def test_generate_compose_hf_dataset_registered(self, tool_names):
        assert "generate.compose_hf_dataset" in tool_names

    def test_generate_compose_hf_dataset_async_registered(self, tool_names):
        assert "generate.compose_hf_dataset_async" in tool_names

    def test_generate_hf_blend_job_status_registered(self, tool_names):
        assert "generate.hf_blend_job_status" in tool_names

    def test_generate_delete_hf_blend_job_registered(self, tool_names):
        assert "generate.delete_hf_blend_job" in tool_names

    def test_normalize_remap_fields_registered(self, tool_names):
        assert "normalize.remap_fields" in tool_names

    # System tools
    def test_system_setup_check_registered(self, tool_names):
        assert "system.setup_check" in tool_names

    def test_system_config_registered(self, tool_names):
        assert "system.config" in tool_names

    def test_system_clear_gpu_cache_registered(self, tool_names):
        assert "system.clear_gpu_cache" in tool_names

    def test_system_set_runtime_env_registered(self, tool_names):
        assert "system.set_runtime_env" in tool_names

    # Finetune tools
    def test_finetune_merge_adapter_registered(self, tool_names):
        assert "finetune.merge_adapter" in tool_names

    def test_finetune_export_gguf_registered(self, tool_names):
        assert "finetune.export_gguf" in tool_names

    def test_finetune_train_vlm_async_registered(self, tool_names):
        assert "finetune.train_vlm_async" in tool_names

    def test_finetune_train_vlm_registered(self, tool_names):
        assert "finetune.train_vlm" in tool_names

    def test_finetune_delete_job_registered(self, tool_names):
        assert "finetune.delete_job" in tool_names

    def test_workflow_delete_job_registered(self, tool_names):
        assert "workflow.delete_job" in tool_names

    def test_workflow_benchmark_finetuning_registered(self, tool_names):
        assert "workflow.benchmark_finetuning" in tool_names

    def test_test_vlm_inference_registered(self, tool_names):
        assert "test.vlm_inference" in tool_names

    def test_validate_preference_dataset_registered(self, tool_names):
        assert "validate.preference_dataset" in tool_names

    def test_validate_composition_registered(self, tool_names):
        assert "validate.composition" in tool_names

    # Judge tools
    def test_judge_evaluate_vlm_registered(self, tool_names):
        assert "judge.evaluate_vlm" in tool_names

    def test_judge_compare_vlm_registered(self, tool_names):
        assert "judge.compare_vlm" in tool_names

    def test_judge_evaluate_vlm_batch_registered(self, tool_names):
        assert "judge.evaluate_vlm_batch" in tool_names

    # Host tools
    def test_host_health_registered(self, tool_names):
        assert "host.health" in tool_names

    def test_host_deploy_vlm_mcp_registered(self, tool_names):
        assert "host.deploy_vlm_mcp" in tool_names

    def test_host_deploy_vlm_api_registered(self, tool_names):
        assert "host.deploy_vlm_api" in tool_names

    def test_host_chat_vlm_registered(self, tool_names):
        assert "host.chat_vlm" in tool_names

    # Total tool count increased
    def test_minimum_tool_count(self, tool_names):
        assert len(tool_names) >= 93


def test_finetune_train_schema_includes_optional_defaults():
    gateway = _make_gateway()
    schema = gateway.mcp._tools["finetune.train"]["schema"]
    props = schema["properties"]

    assert props["num_epochs"]["default"] == 3
    assert props["use_lora"]["default"] is True
    assert props["lora_r"]["default"] == 16
    assert props["lora_alpha"]["default"] == 32
    assert props["lora_dropout"]["default"] == 0.05
    assert props["learning_rate"]["default"] == 2e-4
    assert props["per_device_train_batch_size"]["default"] == 1
    assert props["gradient_accumulation_steps"]["default"] == 4
    assert props["deploy"]["default"] is False
    assert props["special_tokens"]["type"] == "array"
    assert props["thinking_mode"]["default"] == "default"
    assert props["thinking_mode"]["enum"] == ["default", "on", "off"]
    assert "default" not in props["base_model"]
    assert "default" not in props["push_to_hub"]


def test_finetune_async_schema_includes_optional_defaults():
    gateway = _make_gateway()
    schema = gateway.mcp._tools["finetune.train_grpo_async"]["schema"]
    props = schema["properties"]

    assert props["num_epochs"]["default"] == 3
    assert props["use_lora"]["default"] is True
    assert props["lora_r"]["default"] == 16
    assert props["lora_alpha"]["default"] == 32
    assert props["num_generations"]["default"] == 4
    assert props["max_prompt_length"]["default"] == 512
    assert props["max_completion_length"]["default"] == 256
    assert props["per_device_train_batch_size"]["default"] == 1
    assert props["gradient_accumulation_steps"]["default"] == 4
    assert "generation_batch_size" in props
    assert props["load_in_4bit"]["default"] is True
    assert "default" not in props["resume_from_checkpoint"]


def test_preference_training_schemas_accept_adapter_path():
    gateway = _make_gateway()

    dpo_props = gateway.mcp._tools["finetune.train_dpo"]["schema"]["properties"]
    grpo_props = gateway.mcp._tools["finetune.train_grpo"]["schema"]["properties"]
    kto_props = gateway.mcp._tools["finetune.train_kto"]["schema"]["properties"]
    dpo_async_props = gateway.mcp._tools["finetune.train_dpo_async"]["schema"]["properties"]
    grpo_async_props = gateway.mcp._tools["finetune.train_grpo_async"]["schema"]["properties"]
    kto_async_props = gateway.mcp._tools["finetune.train_kto_async"]["schema"]["properties"]

    assert "adapter_path" in dpo_props
    assert "adapter_path" in grpo_props
    assert "adapter_path" in kto_props
    assert "adapter_path" in dpo_async_props
    assert "adapter_path" in grpo_async_props
    assert "adapter_path" in kto_async_props
    assert dpo_props["max_prompt_length"]["default"] == 384
    assert dpo_props["max_length"]["default"] == 512
    assert dpo_async_props["max_prompt_length"]["default"] == 384
    assert dpo_async_props["max_length"]["default"] == 512
    assert dpo_props["auto_tune_defaults"]["default"] is True
    assert grpo_props["auto_tune_defaults"]["default"] is True
    assert kto_props["auto_tune_defaults"]["default"] is True
    assert dpo_async_props["auto_tune_defaults"]["default"] is True
    assert grpo_async_props["auto_tune_defaults"]["default"] is True
    assert kto_async_props["auto_tune_defaults"]["default"] is True


def test_full_pipeline_schemas_accept_adapter_path():
    gateway = _make_gateway()

    full_props = gateway.mcp._tools["workflow.full_pipeline"]["schema"]["properties"]
    full_async_props = gateway.mcp._tools["workflow.full_pipeline_async"]["schema"]["properties"]

    assert "adapter_path" in full_props
    assert "adapter_path" in full_async_props


def test_test_inference_schema_includes_temperature_and_adapter():
    gateway = _make_gateway()
    schema = gateway.mcp._tools["test.inference"]["schema"]
    props = schema["properties"]

    assert props["max_new_tokens"]["default"] == 512
    assert props["temperature"]["default"] == 0.7
    assert props["top_p"]["default"] == 0.9
    assert props["top_k"]["default"] == 50
    assert "adapter_path" in props
    assert props["thinking_mode"]["default"] == "default"
    assert props["thinking_mode"]["enum"] == ["default", "on", "off"]


def test_host_deploy_schema_accepts_system_prompt():
    gateway = _make_gateway()
    schema = gateway.mcp._tools["host.deploy_mcp"]["schema"]
    props = schema["properties"]

    assert "system_prompt" in props
    assert props["system_prompt"]["type"] == "string"
    assert props["thinking_mode"]["default"] == "default"
    assert props["thinking_mode"]["enum"] == ["default", "on", "off"]


def test_curriculum_schema_accepts_staged_inputs():
    gateway = _make_gateway()
    schema = gateway.mcp._tools["finetune.train_curriculum"]["schema"]
    props = schema["properties"]

    assert "stage_dataset_paths" in props
    assert props["stage_dataset_paths"]["type"] == "array"
    assert "stage_training_overrides" in props
    assert props["stage_training_overrides"]["type"] == "array"
    assert props["lora_stage_transition"]["default"] == "continue_adapter"


def test_benchmark_finetuning_schema_accepts_eval_packs_and_reference_models():
    gateway = _make_gateway()
    schema = gateway.mcp._tools["workflow.benchmark_finetuning"]["schema"]
    props = schema["properties"]

    assert "train_dataset_path" in props
    assert "stage_dataset_paths" in props
    assert props["stage_dataset_paths"]["type"] == "array"
    assert "holdout_data_path" in props
    assert "safety_data_path" in props
    assert "training_methods" in props
    assert props["training_methods"]["type"] == "array"
    assert "reference_models" in props
    assert props["reference_models"]["type"] == "array"
    assert "benchmark_gates" in props
    assert props["benchmark_gates"]["type"] == "array"
    assert "eval_file_path" in props
    assert "lora_dropout" in props
    assert "weight_decay" in props
    assert "save_best_model" in props
    assert props["primary_metric"]["default"] == "avg_composite_score"


def test_validate_preference_dataset_schema_includes_sampling_controls():
    gateway = _make_gateway()
    schema = gateway.mcp._tools["validate.preference_dataset"]["schema"]
    props = schema["properties"]

    assert props["technique"]["enum"] == ["dpo", "grpo", "kto"]
    assert props["max_rows"]["default"] == 2000
    assert props["top_k"]["default"] == 5


@pytest.mark.asyncio
async def test_finetune_train_tool_passes_artifact_context():
    gateway = _make_gateway()
    gateway._finetuning_svc = AsyncMock()
    gateway._finetuning_svc.load_dataset_from_file = AsyncMock(
        return_value={
            "success": True,
            "dataset_object": [{"prompt": "hi", "response": "hello"}],
        }
    )
    gateway._finetuning_svc.train_model = AsyncMock(
        return_value={"success": True, "model_path": "/tmp/out"}
    )

    tool = gateway.mcp._tools["finetune.train"]["func"]
    payload = json.loads(
        await tool(dataset_path="/train.jsonl", output_dir="/tmp/out")
    )

    assert payload["success"] is True
    train_kwargs = gateway._finetuning_svc.train_model.await_args.kwargs
    assert train_kwargs["dataset_path"] == "/train.jsonl"
    assert train_kwargs["run_source"] == "finetune.train"


@pytest.mark.asyncio
async def test_validate_schema_accepts_vlm_technique(tmp_path):
    gateway = _make_gateway()
    validate_schema = gateway.mcp._tools["validate.schema"]["func"]
    dataset_path = tmp_path / "vlm.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_path", "image_path": "uploads/images/example.png"},
                            {"type": "text", "text": "Describe this image."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "A short description."}],
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = json.loads(
        await validate_schema(dataset_path=str(dataset_path), technique="vlm_sft")
    )

    assert result["success"] is True
    assert result["technique_requested"] == "vlm_sft"


@pytest.mark.asyncio
async def test_validate_schema_accepts_text_messages_sft(tmp_path):
    gateway = _make_gateway()
    validate_schema = gateway.mcp._tools["validate.schema"]["func"]
    dataset_path = tmp_path / "messages_sft.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Check the docs."},
                    {
                        "role": "assistant",
                        "content": "Searching.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": {"query": "docs"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "search_docs",
                        "content": "Use split-server examples.",
                    },
                    {"role": "assistant", "content": "Use split-server examples."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = json.loads(await validate_schema(dataset_path=str(dataset_path), technique="sft"))

    assert result["success"] is True
    assert result["technique_detected"] == "sft"


@pytest.mark.asyncio
async def test_validate_preference_dataset_returns_dpo_warnings(tmp_path):
    gateway = _make_gateway()
    validate_preference = gateway.mcp._tools["validate.preference_dataset"]["func"]
    dataset_path = tmp_path / "dpo.jsonl"
    dataset_path.write_text(
        (
            json.dumps({
                "prompt": "What is the product?",
                "chosen": "It is a shared WhatsApp sales workspace.",
                "rejected": "I do not know.",
            })
            + "\n"
            + json.dumps({
                "prompt": "How does it help teams?",
                "chosen": "It keeps follow-up and routing in one place.",
                "rejected": "I do not know.",
            })
            + "\n"
        ),
        encoding="utf-8",
    )

    result = json.loads(await validate_preference(dataset_path=str(dataset_path), technique="dpo"))

    assert result["success"] is True
    assert result["technique_analyzed"] == "dpo"
    assert result["status"] == "warn"
    assert result["dpo"]["rejected_stats"]["unique_count"] == 1
    assert result["dpo"]["dominant_rejected_count"] == 2
    assert "hard_negative_ratio" in result["dpo"]
    assert result["guidance"]["starting_recipe"]["start_from_sft_checkpoint"] is True


@pytest.mark.asyncio
async def test_dataset_list_discovers_workspace_data_dataset_even_when_cwd_differs(tmp_path, monkeypatch):
    gateway = _make_gateway()
    dataset_list = gateway.mcp._tools["dataset.list"]["func"]
    dataset_path = tmp_path / "data" / "wa_sales" / "whatsapp_sales_agent_train_expanded.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(
            {
                "system": "You are concise.",
                "user": "What is Salestify?",
                "assistant": "A WhatsApp sales workspace.",
            }
        ) + "\n",
        encoding="utf-8",
    )
    foreign_cwd = tmp_path / "elsewhere"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)

    import mcp_gateway as gateway_module
    monkeypatch.setattr(gateway_module, "__file__", str(tmp_path / "mcp_gateway.py"))

    result = json.loads(await dataset_list())

    assert result["success"] is True
    assert result["scan_roots"] == [str((tmp_path / "data").resolve())]
    assert any(item["file_path"] == str(dataset_path.resolve()) for item in result["datasets"])


@pytest.mark.asyncio
async def test_dataset_list_skips_non_dataset_json_artifacts(tmp_path, monkeypatch):
    gateway = _make_gateway()
    dataset_list = gateway.mcp._tools["dataset.list"]["func"]
    dataset_path = tmp_path / "data" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps({"instruction": "q", "input": "", "output": "a"}) + "\n",
        encoding="utf-8",
    )
    artifact_paths = [
        tmp_path / "output" / "run" / "adapter_config.json",
        tmp_path / "output" / "run" / "tokenizer.json",
        tmp_path / "output" / "run" / "trainer_state.json",
        tmp_path / "output" / "run" / "summary.json",
    ]
    artifact_paths[0].parent.mkdir(parents=True, exist_ok=True)
    artifact_paths[0].write_text(
        json.dumps({"base_model_name_or_path": "demo-model"}),
        encoding="utf-8",
    )
    artifact_paths[1].write_text(
        json.dumps({"model": {"vocab": []}}),
        encoding="utf-8",
    )
    artifact_paths[2].write_text(
        json.dumps({"best_metric": 0.9}),
        encoding="utf-8",
    )
    artifact_paths[3].write_text(
        json.dumps({"summary": "not a dataset"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    import mcp_gateway as gateway_module
    monkeypatch.setattr(gateway_module, "__file__", str(tmp_path / "mcp_gateway.py"))

    result = json.loads(await dataset_list())

    assert result["success"] is True
    paths = {item["file_path"] for item in result["datasets"]}
    assert str(dataset_path.resolve()) in paths
    assert all(str(path.resolve()) not in paths for path in artifact_paths)


@pytest.mark.asyncio
async def test_dataset_list_skips_training_progress_snapshots_in_scan_root(tmp_path, monkeypatch):
    gateway = _make_gateway()
    dataset_list = gateway.mcp._tools["dataset.list"]["func"]
    dataset_path = tmp_path / "data" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps({"instruction": "q", "input": "", "output": "a"}) + "\n",
        encoding="utf-8",
    )
    progress_path = tmp_path / "data" / ".training_progress.jsonl"
    progress_path.write_text(
        json.dumps({"job_id": "job-1", "current_step": 1, "timestamp": "2026-04-09T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    import mcp_gateway as gateway_module
    monkeypatch.setattr(gateway_module, "__file__", str(tmp_path / "mcp_gateway.py"))

    result = json.loads(await dataset_list())

    assert result["success"] is True
    paths = {item["file_path"] for item in result["datasets"]}
    assert str(dataset_path.resolve()) in paths
    assert str(progress_path.resolve()) not in paths


@pytest.mark.asyncio
async def test_dataset_list_excludes_persisted_records_outside_scan_roots(tmp_path, monkeypatch):
    gateway = _make_gateway()
    dataset_list = gateway.mcp._tools["dataset.list"]["func"]
    dataset_path = tmp_path / "data" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps({"instruction": "q", "input": "", "output": "a"}) + "\n",
        encoding="utf-8",
    )
    progress_path = tmp_path / "output" / "run" / ".training_progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps({"job_id": "job-1", "current_step": 1, "timestamp": "2026-04-09T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    gateway._persistence.list_datasets = AsyncMock(return_value=[
        {
            "dataset_id": "train",
            "file_path": str(dataset_path.resolve()),
            "format": "jsonl",
            "row_count": 1,
            "columns": ["instruction", "input", "output"],
            "size_bytes": dataset_path.stat().st_size,
        },
        {
            "dataset_id": ".training_progress",
            "file_path": str(progress_path.resolve()),
            "format": "jsonl",
            "row_count": 1,
            "columns": ["job_id", "current_step", "timestamp"],
            "size_bytes": progress_path.stat().st_size,
        },
    ])
    monkeypatch.chdir(tmp_path)
    import mcp_gateway as gateway_module
    monkeypatch.setattr(gateway_module, "__file__", str(tmp_path / "mcp_gateway.py"))

    result = json.loads(await dataset_list())

    assert result["success"] is True
    paths = {item["file_path"] for item in result["datasets"]}
    assert str(dataset_path.resolve()) in paths
    assert str(progress_path.resolve()) not in paths


@pytest.mark.asyncio
async def test_dataset_list_honors_custom_scan_roots(tmp_path, monkeypatch):
    gateway = _make_gateway()
    dataset_list = gateway.mcp._tools["dataset.list"]["func"]
    dataset_path = tmp_path / "custom-library" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps({"instruction": "q", "input": "", "output": "a"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    import mcp_gateway as gateway_module
    monkeypatch.setattr(gateway_module, "__file__", str(tmp_path / "mcp_gateway.py"))

    result = json.loads(await dataset_list(scan_roots=["custom-library"]))

    assert result["success"] is True
    assert result["scan_roots"]
    assert any(item["file_path"] == str(dataset_path.resolve()) for item in result["datasets"])


@pytest.mark.asyncio
async def test_dataset_list_prunes_stale_persisted_records(tmp_path, monkeypatch):
    gateway = _make_gateway()
    dataset_list = gateway.mcp._tools["dataset.list"]["func"]
    stale_dir = tmp_path / "isolated-library"
    stale_path = str((stale_dir / "ghost.jsonl").resolve())
    gateway._persistence.list_datasets = AsyncMock(return_value=[
        {
            "dataset_id": "ghost",
            "file_path": stale_path,
            "format": "jsonl",
            "row_count": 1,
            "columns": ["instruction", "output"],
            "size_bytes": 12,
        }
    ])
    gateway._persistence.mark_dataset_deleted = AsyncMock(return_value=True)
    monkeypatch.chdir(tmp_path)

    result = json.loads(await dataset_list(scan_roots=[str(stale_dir)]))

    assert result["success"] is True
    assert result["datasets"] == []
    assert result["pruned_stale_records"] == 1
    gateway._persistence.mark_dataset_deleted.assert_awaited_once_with(stale_path)


@pytest.mark.asyncio
async def test_normalize_remap_fields_converts_chat_rows():
    gateway = _make_gateway()
    remap_fields = gateway.mcp._tools["normalize.remap_fields"]["func"]

    result = json.loads(
        await remap_fields(
            data_points=[
                {
                    "system": "You are concise.",
                    "user": "What is Salestify?",
                    "assistant": "A WhatsApp sales workspace.",
                }
            ],
            preset="chat_triplet_to_sft",
        )
    )

    assert result["success"] is True
    assert result["target_format"] == "sft"
    assert result["data_points"] == [
        {
            "instruction": "System: You are concise.\n\nUser: What is Salestify?",
            "input": "",
            "output": "A WhatsApp sales workspace.",
        }
    ]


@pytest.mark.asyncio
async def test_generate_list_profiles_filters_by_mode():
    gateway = _make_gateway()
    list_profiles = gateway.mcp._tools["generate.list_profiles"]["func"]

    result = json.loads(await list_profiles(mode="general"))

    assert result["success"] is True
    assert result["count"] >= 1
    assert all(profile["mode"] == "general" for profile in result["profiles"])


@pytest.mark.asyncio
async def test_generate_get_profile_returns_built_in_profile():
    gateway = _make_gateway()
    get_profile = gateway.mcp._tools["generate.get_profile"]["func"]

    result = json.loads(await get_profile(profile_name="coding_assistant"))

    assert result["success"] is True
    assert result["profile_name"] == "coding_assistant"
    assert result["profile"]["mode"] == "coding"
    assert sum(
        item["weight_percent"] for item in result["profile"]["capability_targets"]
    ) == 100


@pytest.mark.asyncio
async def test_generate_preview_composition_returns_row_plan_and_schema_adapter(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nMulti-hop source material across sections.\n\n"
        "## More\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )

    result = json.loads(
        await preview(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            row_target=25,
        )
    )

    assert result["success"] is True
    assert result["profile_name"] == "general_instruction"
    assert result["objective"] == "sft"
    assert result["schema_adapter"]["name"] == "instruction_input_output"
    assert result["source_totals"]["estimated_chunks"] >= 2
    assert sum(result["resolved_mix"].values()) == 100
    assert sum(result["row_plan"].values()) == 25


@pytest.mark.asyncio
async def test_generate_preview_composition_warns_when_weights_need_normalization(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nSecond chunk.\n",
        encoding="utf-8",
    )

    result = json.loads(
        await preview(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            row_target=20,
            capability_overrides={"multi_hop": 40},
        )
    )

    assert result["success"] is True
    assert any("normalized to 100" in warning.lower() for warning in result["warnings"])


@pytest.mark.asyncio
async def test_generate_preview_composition_warns_for_multi_hop_with_single_chunk(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "single.txt"
    source_path.write_text("Single chunk source only.", encoding="utf-8")

    result = json.loads(
        await preview(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            row_target=10,
        )
    )

    assert result["success"] is True
    assert any("multi-hop" in warning.lower() for warning in result["warnings"])


@pytest.mark.asyncio
async def test_generate_preview_composition_for_coding_profile_detects_code_sources(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "service.py"
    source_path.write_text(
        "def normalize_name(value: str) -> str:\n"
        "    if not value:\n"
        "        return ''\n"
        "    return value.strip().lower()\n",
        encoding="utf-8",
    )

    result = json.loads(
        await preview(
            profile_name="coding_assistant",
            source_paths=[str(source_path)],
            row_target=12,
        )
    )

    assert result["success"] is True
    assert result["profile_name"] == "coding_assistant"
    assert result["mode"] == "coding"
    assert result["objective"] == "sft"
    assert result["source_totals"]["code_files"] == 1
    assert sum(result["row_plan"].values()) == 12


@pytest.mark.asyncio
async def test_generate_preview_composition_for_agent_profile_returns_multi_step_plan(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "tool_notes.md"
    source_path.write_text(
        "# Tool policy\n\nUse `search_docs` when the answer depends on repository docs.\n\n"
        "## Recovery\n\nIf the tool returns no result, retry with a narrower query or stop and say the source was insufficient.\n\n"
        "## Final answer\n\nBase the response only on tool output.\n",
        encoding="utf-8",
    )

    result = json.loads(
        await preview(
            profile_name="agent_tool_calling",
            source_paths=[str(source_path)],
            row_target=9,
        )
    )

    assert result["success"] is True
    assert result["profile_name"] == "agent_tool_calling"
    assert result["mode"] == "agent"
    assert result["objective"] == "sft"
    assert result["schema_adapter"]["name"] == "text_messages"
    assert sum(result["row_plan"].values()) == 9
    assert "multi_step_state" in result["row_plan"]


@pytest.mark.asyncio
async def test_generate_preview_composition_for_dpo_uses_preference_adapter(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )

    result = json.loads(
        await preview(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            row_target=6,
            objective="dpo",
        )
    )

    assert result["success"] is True
    assert result["objective"] == "dpo"
    assert result["schema_adapter"]["name"] == "prompt_chosen_rejected"


@pytest.mark.asyncio
async def test_generate_preview_composition_for_grpo_uses_reward_group_adapter(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )

    result = json.loads(
        await preview(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            row_target=6,
            objective="grpo",
        )
    )

    assert result["success"] is True
    assert result["objective"] == "grpo"
    assert result["schema_adapter"]["name"] == "prompt_responses_rewards"


@pytest.mark.asyncio
async def test_generate_preview_composition_for_kto_uses_binary_label_adapter(tmp_path):
    gateway = _make_gateway()
    preview = gateway.mcp._tools["generate.preview_composition"]["func"]
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )

    result = json.loads(
        await preview(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            row_target=5,
            objective="kto",
        )
    )

    assert result["success"] is True
    assert result["objective"] == "kto"
    assert result["schema_adapter"]["name"] == "prompt_completion_label"
    assert any("binary-label rows" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_generate_register_schema_adapter_lists_runtime_adapter():
    gateway = _make_gateway()
    register_adapter = gateway.mcp._tools["generate.register_schema_adapter"]["func"]
    list_adapters = gateway.mcp._tools["generate.list_schema_adapters"]["func"]
    adapter_name = f"runtime_adapter_{uuid.uuid4().hex[:8]}"

    created = json.loads(
        await register_adapter(
            name=adapter_name,
            canonical_kind="text_sft",
            field_map={"instruction": "prompt", "output": "answer"},
            defaults={"input": ""},
            description="Runtime adapter for gateway tests.",
        )
    )
    listed = json.loads(await list_adapters(canonical_kind="text_sft"))

    assert created["success"] is True
    assert created["schema_adapter"]["name"] == adapter_name
    assert any(
        adapter["name"] == adapter_name for adapter in listed["schema_adapters"]
    )


@pytest.mark.asyncio
async def test_generate_compose_profiled_dataset_writes_manifest_and_validates(tmp_path):
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_profiled_dataset"]["func"]
    validate_composition = gateway.mcp._tools["validate.composition"]["func"]
    gateway._generator_svc = AsyncMock()
    counter = {"n": 0}

    async def _fake_generate_from_page(**kwargs):
        counter["n"] += 1
        return {
            "success": True,
            "data_points": [
                {
                    "instruction": f"Question {counter['n']}",
                    "input": "Direct quote from source.",
                    "output": f"Answer {counter['n']}",
                    "id": counter["n"],
                    "file_name": kwargs["file_name"],
                    "page": kwargs["page_index"] + 1,
                    "text": kwargs["page_text"],
                }
            ],
        }

    gateway._generator_svc.generate_from_page = AsyncMock(side_effect=_fake_generate_from_page)

    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nMulti-hop source material across sections.\n\n"
        "## More\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "profiled.jsonl"

    result = json.loads(
        await compose(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            output_path=str(output_path),
            row_target=6,
        )
    )

    assert result["success"] is True
    assert Path(result["dataset"]["file_path"]).exists()
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile_name"] == "general_instruction"
    assert manifest["row_count"] == 6
    assert manifest["schema_adapter_name"] == "instruction_input_output"

    validation = json.loads(
        await validate_composition(dataset_path=result["dataset"]["file_path"])
    )

    assert validation["success"] is True
    assert validation["status"] in {"pass", "warn"}
    assert validation["profile_name"] == "general_instruction"
    assert validation["capability_counts"]


@pytest.mark.asyncio
async def test_generate_compose_profiled_dataset_supports_coding_profile(tmp_path):
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_profiled_dataset"]["func"]
    validate_composition = gateway.mcp._tools["validate.composition"]["func"]
    gateway._generator_svc = AsyncMock()
    counter = {"n": 0}

    async def _fake_generate_from_page(**kwargs):
        counter["n"] += 1
        return {
            "success": True,
            "data_points": [
                {
                    "instruction": f"Review code path {counter['n']}",
                    "input": "def normalize_name(value: str) -> str:",
                    "output": f"Grounded coding answer {counter['n']}",
                    "id": counter["n"],
                    "file_name": kwargs["file_name"],
                    "page": kwargs["page_index"] + 1,
                    "text": kwargs["page_text"],
                }
            ],
        }

    gateway._generator_svc.generate_from_page = AsyncMock(side_effect=_fake_generate_from_page)

    source_path = tmp_path / "service.py"
    source_path.write_text(
        "def normalize_name(value: str) -> str:\n"
        "    if not value:\n"
        "        return ''\n"
        "    return value.strip().lower()\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "coding_profiled.jsonl"

    result = json.loads(
        await compose(
            profile_name="coding_assistant",
            source_paths=[str(source_path)],
            output_path=str(output_path),
            row_target=5,
        )
    )

    assert result["success"] is True
    assert result["profile_name"] == "coding_assistant"
    assert Path(result["dataset"]["file_path"]).exists()

    validation = json.loads(
        await validate_composition(dataset_path=result["dataset"]["file_path"])
    )

    assert validation["success"] is True
    assert validation["profile_name"] == "coding_assistant"
    assert validation["mode"] == "coding"
    assert validation["capability_counts"]


@pytest.mark.asyncio
async def test_generate_compose_profiled_dataset_supports_agent_profile(tmp_path):
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_profiled_dataset"]["func"]
    validate_composition = gateway.mcp._tools["validate.composition"]["func"]
    dataset_preview = gateway.mcp._tools["dataset.preview"]["func"]
    gateway._generator_svc = AsyncMock()
    counter = {"n": 0}

    async def _fake_generate_from_page(**kwargs):
        counter["n"] += 1
        return {
            "success": True,
            "data_points": [
                {
                    "instruction": f"Choose the next tool step {counter['n']}",
                    "input": "Use `search_docs` when the answer depends on repository docs.",
                    "output": f"Step 1: search_docs({{'query': 'repo docs'}})\nFinal answer: grounded output {counter['n']}",
                    "id": counter["n"],
                    "file_name": kwargs["file_name"],
                    "page": kwargs["page_index"] + 1,
                    "text": kwargs["page_text"],
                }
            ],
        }

    gateway._generator_svc.generate_from_page = AsyncMock(side_effect=_fake_generate_from_page)

    source_path = tmp_path / "agent_tool_notes.md"
    source_path.write_text(
        "# Tool policy\n\nUse `search_docs` when the answer depends on repository docs.\n\n"
        "## Recovery\n\nIf the tool returns no result, retry with a narrower query.\n\n"
        "## Final answer\n\nBase the response only on tool output.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "agent_profiled.jsonl"

    result = json.loads(
        await compose(
            profile_name="agent_tool_calling",
            source_paths=[str(source_path)],
            output_path=str(output_path),
            row_target=4,
        )
    )

    assert result["success"] is True
    assert result["profile_name"] == "agent_tool_calling"
    assert Path(result["dataset"]["file_path"]).exists()

    validation = json.loads(
        await validate_composition(dataset_path=result["dataset"]["file_path"])
    )
    preview = json.loads(await dataset_preview(file_path=result["dataset"]["file_path"], n=1))

    assert validation["success"] is True
    assert validation["profile_name"] == "agent_tool_calling"
    assert validation["mode"] == "agent"
    assert validation["capability_counts"]
    assert preview["rows"][0]["messages"][2]["tool_calls"][0]["function"]["name"] == "search_docs"


@pytest.mark.asyncio
async def test_generate_compose_profiled_dataset_supports_dpo_profile(tmp_path):
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_profiled_dataset"]["func"]
    validate_composition = gateway.mcp._tools["validate.composition"]["func"]
    dataset_preview = gateway.mcp._tools["dataset.preview"]["func"]
    gateway._generator_svc = AsyncMock()
    counter = {"n": 0}

    async def _fake_generate_from_page(**kwargs):
        counter["n"] += 1
        return {
            "success": True,
            "data_points": [
                {
                    "prompt": f"Question {counter['n']}",
                    "chosen": f"Grounded preferred answer {counter['n']}",
                    "rejected": f"Weaker answer {counter['n']}",
                    "id": counter["n"],
                    "file_name": kwargs["file_name"],
                    "page": kwargs["page_index"] + 1,
                    "text": kwargs["page_text"],
                }
            ],
        }

    gateway._generator_svc.generate_from_page = AsyncMock(side_effect=_fake_generate_from_page)

    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "profiled_dpo.jsonl"

    result = json.loads(
        await compose(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            output_path=str(output_path),
            row_target=4,
            objective="dpo",
        )
    )
    validation = json.loads(
        await validate_composition(dataset_path=result["dataset"]["file_path"])
    )
    preview = json.loads(await dataset_preview(file_path=result["dataset"]["file_path"], n=1))

    assert result["success"] is True
    assert result["objective"] == "dpo"
    assert preview["rows"][0]["chosen"].startswith("Grounded preferred answer")
    assert validation["success"] is True
    assert validation["profile_name"] == "general_instruction"


@pytest.mark.asyncio
async def test_generate_compose_profiled_dataset_supports_grpo_profile(tmp_path):
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_profiled_dataset"]["func"]
    validate_composition = gateway.mcp._tools["validate.composition"]["func"]
    dataset_preview = gateway.mcp._tools["dataset.preview"]["func"]
    gateway._generator_svc = AsyncMock()
    counter = {"n": 0}

    async def _fake_generate_from_page(**kwargs):
        counter["n"] += 1
        return {
            "success": True,
            "data_points": [
                {
                    "prompt": f"Question {counter['n']}",
                    "responses": [
                        f"Best answer {counter['n']}",
                        f"Okay answer {counter['n']}",
                        f"Weak answer {counter['n']}",
                    ],
                    "rewards": [1.0, 0.4, -0.2],
                    "id": counter["n"],
                    "file_name": kwargs["file_name"],
                    "page": kwargs["page_index"] + 1,
                    "text": kwargs["page_text"],
                }
            ],
        }

    gateway._generator_svc.generate_from_page = AsyncMock(side_effect=_fake_generate_from_page)

    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "profiled_grpo.jsonl"

    result = json.loads(
        await compose(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            output_path=str(output_path),
            row_target=4,
            objective="grpo",
        )
    )
    validation = json.loads(
        await validate_composition(dataset_path=result["dataset"]["file_path"])
    )
    preview = json.loads(await dataset_preview(file_path=result["dataset"]["file_path"], n=1))

    assert result["success"] is True
    assert result["objective"] == "grpo"
    assert preview["rows"][0]["responses"][0].startswith("Best answer")
    assert preview["rows"][0]["rewards"] == [1.0, 0.4, -0.2]
    assert validation["success"] is True
    assert validation["profile_name"] == "general_instruction"


@pytest.mark.asyncio
async def test_generate_compose_profiled_dataset_supports_kto_profile(tmp_path):
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_profiled_dataset"]["func"]
    validate_composition = gateway.mcp._tools["validate.composition"]["func"]
    dataset_preview = gateway.mcp._tools["dataset.preview"]["func"]
    gateway._generator_svc = AsyncMock()
    counter = {"n": 0}

    async def _fake_generate_from_page(**kwargs):
        counter["n"] += 1
        return {
            "success": True,
            "data_points": [
                {
                    "prompt": f"Question {counter['n']}",
                    "chosen": f"Preferred answer {counter['n']}",
                    "rejected": f"Rejected answer {counter['n']}",
                    "id": counter["n"],
                    "file_name": kwargs["file_name"],
                    "page": kwargs["page_index"] + 1,
                    "text": kwargs["page_text"],
                }
            ],
        }

    gateway._generator_svc.generate_from_page = AsyncMock(side_effect=_fake_generate_from_page)

    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "profiled_kto.jsonl"

    result = json.loads(
        await compose(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            output_path=str(output_path),
            row_target=5,
            objective="kto",
        )
    )
    validation = json.loads(
        await validate_composition(dataset_path=result["dataset"]["file_path"])
    )
    preview = json.loads(await dataset_preview(file_path=result["dataset"]["file_path"], n=5))

    assert result["success"] is True
    assert result["objective"] == "kto"
    labels = {row["label"] for row in preview["rows"]}
    assert labels == {True, False}
    assert all("completion" in row for row in preview["rows"])
    assert validation["success"] is True
    assert validation["profile_name"] == "general_instruction"


@pytest.mark.asyncio
async def test_generate_compose_profiled_dataset_applies_prompt_response_adapter(tmp_path):
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_profiled_dataset"]["func"]
    dataset_preview = gateway.mcp._tools["dataset.preview"]["func"]
    gateway._generator_svc = AsyncMock()

    async def _fake_generate_from_page(**kwargs):
        return {
            "success": True,
            "data_points": [
                {
                    "instruction": "Summarize the source",
                    "input": "Direct quote from source.",
                    "output": "Grounded answer.",
                    "id": 1,
                    "file_name": kwargs["file_name"],
                    "page": kwargs["page_index"] + 1,
                    "text": kwargs["page_text"],
                }
            ],
        }

    gateway._generator_svc.generate_from_page = AsyncMock(side_effect=_fake_generate_from_page)

    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# Intro\n\nGrounded instruction tuning overview.\n\n"
        "## Details\n\nAdditional grounded evidence.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "profiled_prompt_response.jsonl"

    result = json.loads(
        await compose(
            profile_name="general_instruction",
            source_paths=[str(source_path)],
            output_path=str(output_path),
            row_target=2,
            schema_adapter_name="prompt_response",
        )
    )
    preview = json.loads(await dataset_preview(file_path=result["dataset"]["file_path"], n=1))

    assert result["success"] is True
    assert preview["rows"][0]["prompt"]
    assert preview["rows"][0]["response"] == "Grounded answer."
    assert "instruction" not in preview["rows"][0]


@pytest.mark.asyncio
async def test_validate_schema_accepts_prompt_response_sft_rows(tmp_path):
    gateway = _make_gateway()
    validate_schema = gateway.mcp._tools["validate.schema"]["func"]
    dataset_path = tmp_path / "prompt_response.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "Question", "response": "Answer"}) + "\n",
        encoding="utf-8",
    )

    result = json.loads(await validate_schema(dataset_path=str(dataset_path), technique="sft"))

    assert result["success"] is True
    assert result["technique_detected"] == "sft"


@pytest.mark.asyncio
async def test_compose_hf_dataset_normalizes_messages_to_sft_rows():
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_hf_dataset"]["func"]

    class _FakeDataset:
        def __init__(self, rows):
            self._rows = rows
            self.column_names = list(rows[0].keys()) if rows else []

        def __len__(self):
            return len(self._rows)

        def __iter__(self):
            return iter(self._rows)

        def select(self, indices):
            selected = [self._rows[idx] for idx in indices]
            return _FakeDataset(selected)

    fake_rows = [
        {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "chat_template_kwargs": {"style": "chatml"},
        }
    ]

    with patch("datasets.load_dataset", return_value=_FakeDataset(fake_rows)):
        result = json.loads(
            await compose(
                sources=json.dumps(
                    [
                        {
                            "dataset_name": "demo/source",
                            "split": "train",
                            "drop_columns": ["chat_template_kwargs"],
                        }
                    ]
                ),
                target_format="sft",
                shuffle=False,
            )
        )

    assert result["success"] is True
    assert result["count"] == 1
    assert "What is 2+2?" in result["data_points"][0]["prompt"]
    assert "System: Be helpful." in result["data_points"][0]["prompt"]
    assert result["data_points"][0]["response"] == "4"
    assert result["data_points"][0]["_source_dataset"] == "demo/source"


@pytest.mark.asyncio
async def test_compose_hf_dataset_stage2_recipe_uses_published_stage_dataset():
    gateway = _make_gateway()
    compose = gateway.mcp._tools["generate.compose_hf_dataset"]["func"]

    class _FakeDataset:
        def __init__(self, rows):
            self._rows = rows
            self.column_names = list(rows[0].keys()) if rows else []

        def __len__(self):
            return len(self._rows)

        def select(self, indices):
            selected = [self._rows[i] for i in indices]
            return _FakeDataset(selected)

        def __iter__(self):
            return iter(self._rows)

    fake_rows = [{"prompt": "q", "response": "a"}]

    with patch("datasets.load_dataset", return_value=_FakeDataset(fake_rows)) as mock_load_dataset:
        result = json.loads(
            await compose(
                recipe_name="tiny_reasoning_stage_2",
                max_rows_per_source=1,
                shuffle=False,
            )
        )

    assert result["success"] is True
    first_call = mock_load_dataset.call_args_list[0]
    assert first_call.args[0] == "Shekswess/trlm-sft-stage-2-final-2"
    assert first_call.kwargs["split"] == "train"


@pytest.mark.asyncio
async def test_compose_hf_dataset_async_saves_output_and_reports_completed_status():
    gateway = _make_gateway()
    compose_async = gateway.mcp._tools["generate.compose_hf_dataset_async"]["func"]
    job_status = gateway.mcp._tools["generate.hf_blend_job_status"]["func"]

    async def _fake_compose_dataset(**kwargs):
        return {
            "success": True,
            "target_format": "sft",
            "count": 1,
            "per_source_counts": [],
            "data_points": [{"prompt": "q", "response": "a"}],
        }

    async def _fake_save(**kwargs):
        return {
            "success": True,
            "file_path": kwargs["output_path"],
            "row_count": 1,
        }

    with patch.object(gateway.hf_recipe_service, "compose_dataset", side_effect=_fake_compose_dataset), patch.object(
        gateway.dataset_service,
        "save",
        side_effect=_fake_save,
    ):
        started = json.loads(
            await compose_async(
                recipe_name="tiny_reasoning_stage_1",
                output_path="output/tiny_reasoning_stage_1.jsonl",
            )
        )

        assert started["success"] is True
        assert started["status"] == "running"

        payload = None
        for _ in range(20):
            payload = json.loads(await job_status(job_id=started["job_id"]))
            if payload["status"] == "completed":
                break
            await asyncio.sleep(0.05)

    assert payload is not None
    assert payload["status"] == "completed"
    assert payload["result"]["save_result"]["file_path"] == "output/tiny_reasoning_stage_1.jsonl"
