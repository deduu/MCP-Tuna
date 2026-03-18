"""Integration tests for new MCP tools added to the gateway."""

from __future__ import annotations

import json
from unittest.mock import patch

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

    def test_test_vlm_inference_registered(self, tool_names):
        assert "test.vlm_inference" in tool_names

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
        assert len(tool_names) >= 86


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
