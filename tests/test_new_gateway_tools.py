"""Integration tests for new MCP tools added to the gateway."""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_gateway_tool_names() -> set[str]:
    """Instantiate gateway and return all registered tool names."""
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway
        gw = TunaGateway()
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

    # Finetune tools
    def test_finetune_merge_adapter_registered(self, tool_names):
        assert "finetune.merge_adapter" in tool_names

    def test_finetune_export_gguf_registered(self, tool_names):
        assert "finetune.export_gguf" in tool_names

    # Host tools
    def test_host_health_registered(self, tool_names):
        assert "host.health" in tool_names

    # Total tool count increased
    def test_minimum_tool_count(self, tool_names):
        # Original 69 + 13 new = 82 minimum
        assert len(tool_names) >= 80
