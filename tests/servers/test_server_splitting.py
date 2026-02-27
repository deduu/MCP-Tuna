"""Tests for server splitting — verify each standalone server registers the correct tools."""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Data Server
# ---------------------------------------------------------------------------

class TestDataServer:
    @pytest.fixture(scope="class")
    def tool_names(self) -> set[str]:
        from servers.data_server import DataPrepServer
        with patch("shared.provider_factory.create_llm", return_value=None):
            server = DataPrepServer.__new__(DataPrepServer)
            server._config = {}
            server._generator_svc = None
            server._cleaning_svc = None
            server._normalization_svc = None
            server._dataset_svc = None
            from agentsoul.server import MCPServer
            server.mcp = MCPServer("test-data", "1.0.0")
            server._register_tools()
            return set(server.mcp._tools.keys())

    def test_has_extract_tools(self, tool_names):
        assert "extract.load_document" in tool_names

    def test_has_generate_tools(self, tool_names):
        assert "generate.from_document" in tool_names
        assert "generate.from_text" in tool_names
        assert "generate.from_hf_dataset" in tool_names

    def test_has_clean_tools(self, tool_names):
        assert "clean.dataset" in tool_names
        assert "clean.deduplicate" in tool_names

    def test_has_normalize_tools(self, tool_names):
        assert "normalize.dataset" in tool_names
        assert "normalize.standardize_keys" in tool_names

    def test_has_dataset_tools(self, tool_names):
        assert "dataset.save" in tool_names
        assert "dataset.load" in tool_names
        assert "dataset.preview" in tool_names
        assert "dataset.split" in tool_names
        assert "dataset.merge" in tool_names
        assert "dataset.info" in tool_names

    def test_no_finetune_tools(self, tool_names):
        assert not any(t.startswith("finetune.") for t in tool_names)

    def test_no_host_tools(self, tool_names):
        assert not any(t.startswith("host.") for t in tool_names)


# ---------------------------------------------------------------------------
# Model Eval Server
# ---------------------------------------------------------------------------

class TestModelEvalServer:
    @pytest.fixture(scope="class")
    def tool_names(self) -> set[str]:
        from model_evaluator_pipeline.mcp.server import ModelEvalMCPServer
        server = ModelEvalMCPServer.__new__(ModelEvalMCPServer)
        server._eval_svc = None
        server._judge_svc = None
        server._ft_svc = None
        server._eval_config = None
        server._judge_config = None
        server._ft_config = None
        from agentsoul.server import MCPServer
        server.mcp = MCPServer("test-model-eval", "1.0.0")
        server._register_tools()
        return set(server.mcp._tools.keys())

    def test_has_evaluate_model_tools(self, tool_names):
        assert "evaluate_model.single" in tool_names
        assert "evaluate_model.batch" in tool_names
        assert "evaluate_model.export" in tool_names
        assert "evaluate_model.summary" in tool_names

    def test_has_judge_tools(self, tool_names):
        assert "judge.evaluate" in tool_names
        assert "judge.evaluate_multi" in tool_names
        assert "judge.evaluate_batch" in tool_names
        assert "judge.compare_pair" in tool_names
        assert "judge.list_types" in tool_names
        assert "judge.export" in tool_names
        assert "judge.create_rubric" in tool_names

    def test_has_ft_eval_tools(self, tool_names):
        assert "ft_eval.single" in tool_names
        assert "ft_eval.batch" in tool_names
        assert "ft_eval.summary" in tool_names
        assert "ft_eval.export" in tool_names

    def test_no_finetune_tools(self, tool_names):
        assert not any(t.startswith("finetune.") for t in tool_names)


# ---------------------------------------------------------------------------
# Train Server
# ---------------------------------------------------------------------------

class TestTrainServer:
    @pytest.fixture(scope="class")
    def tool_names(self) -> set[str]:
        from servers.train_server import TrainServer
        server = TrainServer.__new__(TrainServer)
        server._config = {}
        server._finetuning_svc = None
        server._job_manager_instance = None
        from agentsoul.server import MCPServer
        server.mcp = MCPServer("test-train", "1.0.0")
        server._register_tools()
        return set(server.mcp._tools.keys())

    def test_has_system_tools(self, tool_names):
        assert "system.check_resources" in tool_names
        assert "system.preflight_check" in tool_names
        assert "system.setup_check" in tool_names

    def test_has_finetune_tools(self, tool_names):
        assert "finetune.train" in tool_names
        assert "finetune.merge_adapter" in tool_names
        assert "finetune.export_gguf" in tool_names
        assert "finetune.load_dataset" in tool_names

    def test_has_test_tools(self, tool_names):
        assert "test.inference" in tool_names
        assert "test.compare_models" in tool_names

    def test_has_validate_tools(self, tool_names):
        assert "validate.model_info" in tool_names
        assert "validate.list_models" in tool_names

    def test_no_generate_tools(self, tool_names):
        assert not any(t.startswith("generate.") for t in tool_names)


# ---------------------------------------------------------------------------
# Host Server
# ---------------------------------------------------------------------------

class TestHostServer:
    @pytest.fixture(scope="class")
    def tool_names(self) -> set[str]:
        from servers.host_server import HostServer
        server = HostServer()
        return set(server.mcp._tools.keys())

    def test_has_host_tools(self, tool_names):
        assert "host.deploy_mcp" in tool_names
        assert "host.deploy_api" in tool_names
        assert "host.list_deployments" in tool_names
        assert "host.stop" in tool_names
        assert "host.health" in tool_names

    def test_tool_count(self, tool_names):
        assert len(tool_names) == 5


# ---------------------------------------------------------------------------
# Eval Server
# ---------------------------------------------------------------------------

class TestEvalServer:
    @pytest.fixture(scope="class")
    def tool_names(self) -> set[str]:
        from servers.eval_server import EvalServer
        server = EvalServer.__new__(EvalServer)
        server._config = {}
        server._inner = None
        from agentsoul.server import MCPServer
        server.mcp = MCPServer("test-eval", "1.0.0")
        server._register_tools()
        return set(server.mcp._tools.keys())

    def test_has_evaluate_tools(self, tool_names):
        assert "evaluate.dataset" in tool_names
        assert "evaluate.filter_by_quality" in tool_names
        assert "evaluate.statistics" in tool_names
        assert "evaluate.list_metrics" in tool_names

    def test_tool_count(self, tool_names):
        assert len(tool_names) == 4

    def test_no_finetune_tools(self, tool_names):
        assert not any(t.startswith("finetune.") for t in tool_names)

    def test_no_host_tools(self, tool_names):
        assert not any(t.startswith("host.") for t in tool_names)


# ---------------------------------------------------------------------------
# Orchestrate Server
# ---------------------------------------------------------------------------

class TestOrchestrateServer:
    @pytest.fixture(scope="class")
    def tool_names(self) -> set[str]:
        from orchestration.mcp.server import OrchestrationMCPServer
        server = OrchestrationMCPServer.__new__(OrchestrationMCPServer)
        server._svc = None
        server._config = {}
        from agentsoul.server import MCPServer
        server.mcp = MCPServer("test-orchestrate", "1.0.0")
        server._register_tools()
        return set(server.mcp._tools.keys())

    def test_has_orchestration_tools(self, tool_names):
        assert "orchestration.generate_problems" in tool_names
        assert "orchestration.collect_trajectories" in tool_names
        assert "orchestration.build_training_data" in tool_names

    def test_tool_count(self, tool_names):
        assert len(tool_names) == 3

    def test_no_finetune_tools(self, tool_names):
        assert not any(t.startswith("finetune.") for t in tool_names)

    def test_no_dataset_tools(self, tool_names):
        assert not any(t.startswith("dataset.") for t in tool_names)
