"""Unit tests for the orchestration training module."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestration.trajectory import (
    TrajectoryRecorder,
    Trajectory,
    TurnRecord,
)
from orchestration.rewards import OrchestrationRewardFunction
from orchestration.orchestration_trainer import (
    OrchestrationDataService,
    _format_tool_descriptions,
)
from shared.config import OrchestrationConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_mock_llm_response(content: str):
    """Create a mock LLM response object."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = None
    resp.usage = {"prompt_tokens": 100, "completion_tokens": 50}
    return resp


async def _fake_agent_run_events(user_input, **kwargs):
    """Simulate the event stream from AgentSoul.run().

    The real agent yields these externally-visible events:
    token, tool_exec_start, tool_exec_end, phase_start, phase_end, complete.
    turn_metadata is NOT exposed to external callers.
    """
    # Turn 1: LLM produces tokens, then tool execution
    yield {"type": "token", "content": "Let me search"}
    yield {"type": "tool_exec_start", "tool": "search_db"}
    await asyncio.sleep(0.01)  # simulate tool latency
    yield {"type": "tool_exec_end", "tool": "search_db"}

    # Turn 2: final answer tokens
    yield {"type": "token", "content": "The answer is 42"}

    # Complete event carries history + usage
    yield {
        "type": "complete",
        "content": "The answer is 42",
        "history": [
            {
                "turn_number": 1,
                "tool_calls": [
                    {"name": "search_db", "arguments": {"q": "meaning"}, "result": "42"}
                ],
            },
            {"turn_number": 2, "tool_calls": []},
        ],
        "usage": {"prompt_tokens": 130, "completion_tokens": 50},
    }


# ──────────────────────────────────────────────
# TrajectoryRecorder tests
# ──────────────────────────────────────────────

class TestTrajectoryRecorder:
    def test_cost_table_defaults(self):
        recorder = TrajectoryRecorder()
        assert "gpt-4o" in recorder.COST_TABLE
        assert "local" in recorder.COST_TABLE
        assert recorder.COST_TABLE["local"]["input"] == 0.0

    def test_cost_table_custom_merge(self):
        recorder = TrajectoryRecorder(cost_table={"claude-opus": {"input": 15.0 / 1e6, "output": 75.0 / 1e6}})
        assert "claude-opus" in recorder.COST_TABLE
        assert "gpt-4o" in recorder.COST_TABLE  # originals still present

    def test_estimate_cost_known_model(self):
        recorder = TrajectoryRecorder()
        cost = recorder._estimate_cost("gpt-4o-mini", {"prompt_tokens": 1000, "completion_tokens": 500})
        expected = 1000 * 0.15 / 1e6 + 500 * 0.60 / 1e6
        assert abs(cost - expected) < 1e-10

    def test_estimate_cost_unknown_model_free(self):
        recorder = TrajectoryRecorder()
        cost = recorder._estimate_cost("some-unknown-model", {"prompt_tokens": 1000, "completion_tokens": 500})
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_record_captures_trajectory(self):
        recorder = TrajectoryRecorder()
        mock_agent = MagicMock()
        mock_agent.model_id = "gpt-4o-mini"
        mock_agent.run = _fake_agent_run_events

        trajectory = await recorder.record(mock_agent, "What is the meaning of life?")

        assert isinstance(trajectory, Trajectory)
        assert trajectory.task == "What is the meaning of life?"
        assert trajectory.model_id == "gpt-4o-mini"
        assert trajectory.final_answer == "The answer is 42"
        assert trajectory.total_latency_s > 0
        assert len(trajectory.turns) == 2
        # First turn should have a tool call
        assert len(trajectory.turns[0].tool_calls) == 1
        assert trajectory.turns[0].tool_calls[0]["name"] == "search_db"
        assert trajectory.turns[0].tool_calls[0]["latency_s"] > 0

    @pytest.mark.asyncio
    async def test_record_merges_history_details(self):
        recorder = TrajectoryRecorder()
        mock_agent = MagicMock()
        mock_agent.model_id = "gpt-4o-mini"
        mock_agent.run = _fake_agent_run_events

        trajectory = await recorder.record(mock_agent, "test")

        # Tool call should have arguments and result merged from history
        tc = trajectory.turns[0].tool_calls[0]
        assert tc["arguments"] == {"q": "meaning"}
        assert tc["result"] == "42"


# ──────────────────────────────────────────────
# TurnRecord & Trajectory tests
# ──────────────────────────────────────────────

class TestDataclasses:
    def test_turn_record_defaults(self):
        tr = TurnRecord(turn_number=1, model_id="gpt-4o")
        assert tr.tool_calls == []
        assert tr.llm_tokens == {"prompt_tokens": 0, "completion_tokens": 0}
        assert tr.turn_latency_s == 0.0
        assert tr.estimated_cost_usd == 0.0

    def test_trajectory_to_dict(self):
        t = Trajectory(task="test", model_id="gpt-4o")
        d = t.to_dict()
        assert d["task"] == "test"
        assert d["model_id"] == "gpt-4o"
        assert isinstance(d["turns"], list)
        assert d["total_cost_usd"] == 0.0


# ──────────────────────────────────────────────
# OrchestrationRewardFunction tests
# ──────────────────────────────────────────────

class TestRewardFunction:
    def test_cost_reward_under_budget(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(task="t", total_cost_usd=0.3, model_id="x")
        assert rf.cost_reward(traj, budget=1.0) == pytest.approx(0.7)

    def test_cost_reward_over_budget(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(task="t", total_cost_usd=2.0, model_id="x")
        assert rf.cost_reward(traj, budget=1.0) == 0.0

    def test_cost_reward_zero_budget(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(task="t", total_cost_usd=0.5, model_id="x")
        assert rf.cost_reward(traj, budget=0.0) == 1.0

    def test_latency_reward_under_budget(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(task="t", total_latency_s=15.0, model_id="x")
        assert rf.latency_reward(traj, time_budget=60.0) == pytest.approx(0.75)

    def test_latency_reward_over_budget(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(task="t", total_latency_s=120.0, model_id="x")
        assert rf.latency_reward(traj, time_budget=60.0) == 0.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_parses_score(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_mock_llm_response('{"score": 0.85}'))
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(task="test task", final_answer="42", model_id="x")
        score = await rf.accuracy_reward(traj, "The answer is 42")
        assert score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_accuracy_reward_handles_bad_json(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_mock_llm_response("not json"))
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(task="test", final_answer="ans", model_id="x")
        score = await rf.accuracy_reward(traj, "truth")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_compute_weighted_score(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_mock_llm_response('{"score": 0.8}'))
        rf = OrchestrationRewardFunction(
            mock_llm,
            weights={"accuracy": 0.5, "cost": 0.25, "latency": 0.25},
        )
        traj = Trajectory(
            task="t", final_answer="a", model_id="x",
            total_cost_usd=0.2, total_latency_s=10.0,
        )
        result = await rf.compute(traj, ground_truth="gt", budget=1.0, time_budget=60.0)

        assert "total" in result
        assert "accuracy" in result
        assert "cost" in result
        assert "latency" in result

        expected_acc = 0.8
        expected_cost = 1.0 - 0.2 / 1.0  # 0.8
        expected_lat = 1.0 - 10.0 / 60.0  # ~0.833
        expected_total = 0.5 * expected_acc + 0.25 * expected_cost + 0.25 * expected_lat
        assert result["total"] == pytest.approx(expected_total, abs=0.01)

    @pytest.mark.asyncio
    async def test_compute_without_ground_truth(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        traj = Trajectory(
            task="t", final_answer="a", model_id="x",
            total_cost_usd=0.0, total_latency_s=0.0,
        )
        result = await rf.compute(traj)
        # No ground truth → accuracy defaults to 0.5
        assert result["accuracy"] == 0.5
        # Zero cost and latency → perfect cost/latency scores
        assert result["cost"] == 1.0
        assert result["latency"] == 1.0


# ──────────────────────────────────────────────
# OrchestrationDataService tests
# ──────────────────────────────────────────────

class TestOrchestrationDataService:
    @pytest.mark.asyncio
    async def test_generate_problems(self):
        problems_json = json.dumps([
            {"task": "Look up order #123", "expected_tools": ["search_db"],
             "difficulty": "easy", "ground_truth_hint": "order details"},
            {"task": "Analyze sentiment of reviews", "expected_tools": ["sentiment_api"],
             "difficulty": "medium", "ground_truth_hint": "positive/negative"},
        ])
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_mock_llm_response(problems_json))
        rf = OrchestrationRewardFunction(mock_llm)
        ods = OrchestrationDataService(mock_llm, rf)

        problems = await ods.generate_problems("customer support", num_problems=2)

        assert len(problems) == 2
        assert problems[0]["task"] == "Look up order #123"
        assert problems[1]["difficulty"] == "medium"

    @pytest.mark.asyncio
    async def test_collect_trajectories(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        ods = OrchestrationDataService(mock_llm, rf)

        mock_agent = MagicMock()
        mock_agent.model_id = "gpt-4o-mini"
        mock_agent.run = _fake_agent_run_events

        problems = [{"task": "find order", "ground_truth_hint": "order info"}]
        collected = await ods.collect_trajectories(
            problems, mock_agent, n_per_problem=2, temperatures=[0.3, 0.7],
        )

        assert len(collected) == 1
        assert len(collected[0]["trajectories"]) == 2
        assert collected[0]["problem"]["task"] == "find order"

    @pytest.mark.asyncio
    async def test_build_training_data_sft(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_mock_llm_response('{"score": 0.9}'))
        rf = OrchestrationRewardFunction(mock_llm)
        ods = OrchestrationDataService(mock_llm, rf)

        collected = [{
            "problem": {"task": "test task", "ground_truth_hint": "answer"},
            "trajectories": [
                {"task": "test task", "turns": [
                    {"tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
                     "turn_number": 1}
                ], "final_answer": "good answer", "total_cost_usd": 0.1,
                 "total_latency_s": 5.0, "total_tokens": 100, "model_id": "m"},
                {"task": "test task", "turns": [],
                 "final_answer": "bad answer", "total_cost_usd": 0.5,
                 "total_latency_s": 30.0, "total_tokens": 200, "model_id": "m"},
            ],
        }]

        data = await ods.build_training_data(collected, format="sft")

        assert len(data) == 1
        assert "instruction" in data[0]
        assert "output" in data[0]
        assert "Available tools" in data[0]["instruction"]
        assert "Task: test task" in data[0]["instruction"]

    @pytest.mark.asyncio
    async def test_build_training_data_dpo(self):
        mock_llm = MagicMock()
        # Return different scores for the two trajectories
        mock_llm.chat = AsyncMock(side_effect=[
            _make_mock_llm_response('{"score": 0.9}'),
            _make_mock_llm_response('{"score": 0.2}'),
        ])
        rf = OrchestrationRewardFunction(mock_llm)
        ods = OrchestrationDataService(mock_llm, rf)

        collected = [{
            "problem": {"task": "test", "ground_truth_hint": "ans"},
            "trajectories": [
                {"task": "test", "turns": [], "final_answer": "good",
                 "total_cost_usd": 0.1, "total_latency_s": 5.0,
                 "total_tokens": 50, "model_id": "m"},
                {"task": "test", "turns": [], "final_answer": "bad",
                 "total_cost_usd": 0.8, "total_latency_s": 50.0,
                 "total_tokens": 200, "model_id": "m"},
            ],
        }]

        data = await ods.build_training_data(collected, format="dpo")

        assert len(data) == 1
        assert "prompt" in data[0]
        assert "chosen" in data[0]
        assert "rejected" in data[0]

    @pytest.mark.asyncio
    async def test_build_training_data_grpo(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_mock_llm_response('{"score": 0.7}'))
        rf = OrchestrationRewardFunction(mock_llm)
        ods = OrchestrationDataService(mock_llm, rf)

        collected = [{
            "problem": {"task": "test", "ground_truth_hint": "ans"},
            "trajectories": [
                {"task": "test", "turns": [], "final_answer": "a1",
                 "total_cost_usd": 0.1, "total_latency_s": 5.0,
                 "total_tokens": 50, "model_id": "m"},
            ],
        }]

        data = await ods.build_training_data(collected, format="grpo")

        assert len(data) == 1
        assert "prompt" in data[0]
        assert "responses" in data[0]
        assert "rewards" in data[0]
        assert isinstance(data[0]["rewards"], list)

    @pytest.mark.asyncio
    async def test_export_jsonl(self):
        mock_llm = MagicMock()
        rf = OrchestrationRewardFunction(mock_llm)
        ods = OrchestrationDataService(mock_llm, rf)

        data = [{"instruction": "do X", "output": "did X"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            result = await ods.export(data, path, file_format="jsonl")

            assert result["success"] is True
            assert result["num_examples"] == 1
            assert os.path.exists(path)

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["instruction"] == "do X"


# ──────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────

class TestOrchestrationConfig:
    def test_defaults(self):
        cfg = OrchestrationConfig()
        assert cfg.num_problems == 50
        assert cfg.n_per_problem == 4
        assert cfg.temperatures == [0.3, 0.5, 0.7, 1.0]
        assert cfg.cost_budget == 1.0
        assert cfg.latency_budget == 60.0
        assert cfg.reward_weights == {"accuracy": 0.5, "cost": 0.25, "latency": 0.25}
        assert cfg.output_format == "sft"
        assert cfg.base_model == "Qwen/Qwen3-1.7B"

    def test_inherits_pipeline_config(self):
        cfg = OrchestrationConfig(model="gpt-4o-mini", debug=True)
        assert cfg.model == "gpt-4o-mini"
        assert cfg.debug is True

    def test_custom_weights(self):
        cfg = OrchestrationConfig(
            reward_weights={"accuracy": 0.8, "cost": 0.1, "latency": 0.1}
        )
        assert cfg.reward_weights["accuracy"] == 0.8


# ──────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────

class TestHelpers:
    def test_format_tool_descriptions_none(self):
        result = _format_tool_descriptions(None)
        assert result == "(no tools provided)"

    def test_format_tool_descriptions_with_tools(self):
        tools = [
            {"name": "search", "description": "Search the database", "parameters": {"query": "str"}},
        ]
        result = _format_tool_descriptions(tools)
        assert "search" in result
        assert "Search the database" in result

    def test_format_tool_descriptions_openai_format(self):
        tools = [{
            "function": {
                "name": "lookup",
                "description": "Look up info",
                "parameters": {"type": "object"},
            }
        }]
        result = _format_tool_descriptions(tools)
        assert "lookup" in result
        assert "Look up info" in result
