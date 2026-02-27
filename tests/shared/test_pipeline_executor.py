"""Tests for shared.pipeline_executor — server-side pipeline execution engine."""
from __future__ import annotations

import json
import pytest

from shared.pipeline_executor import PipelineExecutor, PipelineStep, PipelineResult, StepResult


# ---------------------------------------------------------------------------
# Helpers — mock tool functions
# ---------------------------------------------------------------------------

def _make_tools(**name_to_result: dict) -> dict:
    """Build a mock _tools dict matching gateway.mcp._tools structure.

    Each keyword argument maps a tool name to the dict that the tool returns.
    The mock function serialises the dict to JSON (like real MCP tools do).
    """
    tools = {}
    for name, result_dict in name_to_result.items():
        async def _fn(_result=result_dict, **kwargs):  # noqa: E501
            return json.dumps(_result)
        tools[name] = {"func": _fn, "description": f"mock {name}", "schema": {}}
    return tools


def _make_echo_tool(name: str = "echo") -> dict:
    """A tool that echoes its kwargs back as the result, with success=True."""
    async def _fn(**kwargs):
        return json.dumps({"success": True, **kwargs})
    return {name: {"func": _fn, "description": "echo tool", "schema": {}}}


def _make_failing_tool(name: str = "fail", error: str = "boom") -> dict:
    """A tool that returns success=False."""
    async def _fn(**kwargs):
        return json.dumps({"success": False, "error": error})
    return {name: {"func": _fn, "description": "failing tool", "schema": {}}}


def _make_exception_tool(name: str = "explode") -> dict:
    """A tool that raises an exception."""
    async def _fn(**kwargs):
        raise RuntimeError("unexpected explosion")
    return {name: {"func": _fn, "description": "exploding tool", "schema": {}}}


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.asyncio
    async def test_empty_steps_rejected(self):
        executor = PipelineExecutor(tools={})
        result = await executor.execute(steps=[])
        assert result.success is False
        assert "empty" in result.error.lower() or "at least" in result.error.lower()
        assert result.steps_completed == 0

    @pytest.mark.asyncio
    async def test_invalid_tool_name(self):
        executor = PipelineExecutor(tools={})
        steps = [PipelineStep(tool="nonexistent.tool", params={})]
        result = await executor.execute(steps)
        assert result.success is False
        assert "nonexistent.tool" in result.error

    @pytest.mark.asyncio
    async def test_prev_ref_in_step_0(self):
        tools = _make_tools(**{"a.tool": {"success": True}})
        executor = PipelineExecutor(tools=tools)
        steps = [PipelineStep(tool="a.tool", params={"x": "$prev.model_path"})]
        result = await executor.execute(steps)
        assert result.success is False
        assert "$prev" in result.error

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        tools = _make_tools(**{"a.tool": {"success": True}})
        executor = PipelineExecutor(tools=tools)
        steps = [PipelineStep(tool="a.tool", params={})] * 21
        result = await executor.execute(steps)
        assert result.success is False
        assert "20" in result.error


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------

class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_valid_pipeline(self):
        tools = _make_tools(
            **{"step1": {"success": True, "model_path": "/m"}, "step2": {"success": True}}
        )
        executor = PipelineExecutor(tools=tools)
        steps = [
            PipelineStep(tool="step1", params={}),
            PipelineStep(tool="step2", params={"model": "$prev.model_path"}),
        ]
        result = await executor.execute(steps, dry_run=True)
        assert result.success is True
        assert result.steps_completed == 0  # nothing executed

    @pytest.mark.asyncio
    async def test_dry_run_catches_invalid_tool(self):
        executor = PipelineExecutor(tools={})
        steps = [PipelineStep(tool="bad.tool", params={})]
        result = await executor.execute(steps, dry_run=True)
        assert result.success is False
        assert "bad.tool" in result.error

    @pytest.mark.asyncio
    async def test_dry_run_catches_prev_ref_in_step_0(self):
        tools = _make_tools(**{"a.tool": {"success": True}})
        executor = PipelineExecutor(tools=tools)
        steps = [PipelineStep(tool="a.tool", params={"x": "$prev.key"})]
        result = await executor.execute(steps, dry_run=True)
        assert result.success is False
        assert "$prev" in result.error


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------

class TestExecution:
    @pytest.mark.asyncio
    async def test_single_step_success(self):
        tools = _make_tools(**{"check": {"success": True, "gpu": "RTX 3050"}})
        executor = PipelineExecutor(tools=tools)
        steps = [PipelineStep(tool="check", params={})]
        result = await executor.execute(steps)
        assert result.success is True
        assert result.steps_completed == 1
        assert result.total_steps == 1
        assert result.results[0].success is True
        assert result.results[0].result["gpu"] == "RTX 3050"

    @pytest.mark.asyncio
    async def test_two_step_chaining(self):
        tools = {
            **_make_tools(**{"train": {"success": True, "model_path": "/output/model"}}),
            **_make_echo_tool("infer"),
        }
        executor = PipelineExecutor(tools=tools)
        steps = [
            PipelineStep(tool="train", params={"data": "train.jsonl"}),
            PipelineStep(tool="infer", params={"model": "$prev.model_path"}),
        ]
        result = await executor.execute(steps)
        assert result.success is True
        assert result.steps_completed == 2
        # The echo tool returns its kwargs — verify $prev was resolved
        assert result.results[1].result["model"] == "/output/model"

    @pytest.mark.asyncio
    async def test_stop_on_failure(self):
        tools = {
            **_make_tools(**{"step1": {"success": True, "v": 1}}),
            **_make_failing_tool("step2", error="OOM"),
            **_make_tools(**{"step3": {"success": True, "v": 3}}),
        }
        executor = PipelineExecutor(tools=tools)
        steps = [
            PipelineStep(tool="step1", params={}),
            PipelineStep(tool="step2", params={}),
            PipelineStep(tool="step3", params={}),
        ]
        result = await executor.execute(steps)
        assert result.success is False
        assert result.steps_completed == 1  # only step1 succeeded
        assert result.total_steps == 3
        assert len(result.results) == 2  # step1 + failed step2
        assert "step2" in result.error or "OOM" in result.error

    @pytest.mark.asyncio
    async def test_tool_exception_caught(self):
        tools = _make_exception_tool("explode")
        executor = PipelineExecutor(tools=tools)
        steps = [PipelineStep(tool="explode", params={})]
        result = await executor.execute(steps)
        assert result.success is False
        assert result.steps_completed == 0
        assert "unexpected explosion" in result.error.lower() or "exception" in result.error.lower()


# ---------------------------------------------------------------------------
# Ref resolution tests
# ---------------------------------------------------------------------------

class TestRefResolution:
    @pytest.mark.asyncio
    async def test_prev_ref_missing_key(self):
        tools = {
            **_make_tools(**{"step1": {"success": True, "a": 1}}),
            **_make_echo_tool("step2"),
        }
        executor = PipelineExecutor(tools=tools)
        steps = [
            PipelineStep(tool="step1", params={}),
            PipelineStep(tool="step2", params={"x": "$prev.nonexistent"}),
        ]
        result = await executor.execute(steps)
        assert result.success is False
        assert "nonexistent" in result.error

    @pytest.mark.asyncio
    async def test_nested_prev_ref(self):
        tools = {
            **_make_tools(**{"step1": {"success": True, "path": "/out"}}),
            **_make_echo_tool("step2"),
        }
        executor = PipelineExecutor(tools=tools)
        steps = [
            PipelineStep(tool="step1", params={}),
            PipelineStep(tool="step2", params={"config": {"model": "$prev.path"}}),
        ]
        result = await executor.execute(steps)
        assert result.success is True
        assert result.results[1].result["config"]["model"] == "/out"

    @pytest.mark.asyncio
    async def test_prev_ref_in_list_values(self):
        tools = {
            **_make_tools(**{"step1": {"success": True, "prompt": "Hello"}}),
            **_make_echo_tool("step2"),
        }
        executor = PipelineExecutor(tools=tools)
        steps = [
            PipelineStep(tool="step1", params={}),
            PipelineStep(tool="step2", params={"prompts": ["$prev.prompt", "world"]}),
        ]
        result = await executor.execute(steps)
        assert result.success is True
        assert result.results[1].result["prompts"] == ["Hello", "world"]


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestResultStructure:
    @pytest.mark.asyncio
    async def test_result_is_pydantic_model(self):
        tools = _make_tools(**{"t": {"success": True, "v": 42}})
        executor = PipelineExecutor(tools=tools)
        result = await executor.execute([PipelineStep(tool="t", params={})])
        assert isinstance(result, PipelineResult)
        assert isinstance(result.results[0], StepResult)

    @pytest.mark.asyncio
    async def test_final_result_is_last_success(self):
        tools = _make_tools(
            **{"a": {"success": True, "x": 1}, "b": {"success": True, "y": 2}}
        )
        executor = PipelineExecutor(tools=tools)
        steps = [
            PipelineStep(tool="a", params={}),
            PipelineStep(tool="b", params={}),
        ]
        result = await executor.execute(steps)
        assert result.final_result == {"success": True, "y": 2}

    @pytest.mark.asyncio
    async def test_non_json_tool_result(self):
        async def _bad_fn(**kwargs):
            return "not valid json {{"
        tools = {"bad": {"func": _bad_fn, "description": "bad", "schema": {}}}
        executor = PipelineExecutor(tools=tools)
        result = await executor.execute([PipelineStep(tool="bad", params={})])
        assert result.success is False
        assert result.steps_completed == 0
