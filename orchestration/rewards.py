"""
Multi-objective reward function for orchestration trajectories.

Scores trajectories on accuracy (LLM-as-judge), cost, and latency
with configurable weights.
"""

import json
from typing import Any, Dict, Optional

from orchestration.trajectory import Trajectory


DEFAULT_WEIGHTS = {"accuracy": 0.5, "cost": 0.25, "latency": 0.25}

JUDGE_PROMPT = """You are an impartial evaluator. Score how well the assistant's answer addresses the given task.

Task: {task}

Assistant's Answer:
{answer}

Ground Truth (reference):
{ground_truth}

Return ONLY a JSON object: {{"score": <float between 0.0 and 1.0>}}

Scoring guidelines:
- 1.0: Fully correct and complete answer
- 0.7-0.9: Mostly correct with minor issues
- 0.4-0.6: Partially correct
- 0.1-0.3: Mostly incorrect
- 0.0: Completely wrong or no useful answer"""


class OrchestrationRewardFunction:
    """Multi-objective reward: accuracy + cost + latency."""

    def __init__(self, llm, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            llm: A BaseLLM provider used for LLM-as-judge accuracy scoring.
            weights: Dict with keys 'accuracy', 'cost', 'latency' summing to 1.0.
        """
        self.llm = llm
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    async def accuracy_reward(
        self, trajectory: Trajectory, ground_truth: str
    ) -> float:
        """Score answer quality via LLM-as-judge. Returns float in [0, 1]."""
        prompt = JUDGE_PROMPT.format(
            task=trajectory.task,
            answer=trajectory.final_answer,
            ground_truth=ground_truth,
        )
        response = await self.llm.chat(
            [{"role": "user", "content": prompt}], tools=None
        )
        try:
            parsed = json.loads(response.content)
            return max(0.0, min(1.0, float(parsed["score"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return 0.0

    def cost_reward(self, trajectory: Trajectory, budget: float = 1.0) -> float:
        """Normalized inverse cost. Cheaper trajectories score higher."""
        if budget <= 0:
            return 1.0
        return max(0.0, 1.0 - trajectory.total_cost_usd / budget)

    def latency_reward(
        self, trajectory: Trajectory, time_budget: float = 60.0
    ) -> float:
        """Normalized inverse latency. Faster trajectories score higher."""
        if time_budget <= 0:
            return 1.0
        return max(0.0, 1.0 - trajectory.total_latency_s / time_budget)

    async def compute(
        self,
        trajectory: Trajectory,
        ground_truth: Optional[str] = None,
        budget: float = 1.0,
        time_budget: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Compute weighted reward score.

        Returns:
            Dict with 'total', 'accuracy', 'cost', 'latency' scores.
        """
        acc = (
            await self.accuracy_reward(trajectory, ground_truth)
            if ground_truth
            else 0.5  # neutral when no ground truth
        )
        cost = self.cost_reward(trajectory, budget)
        latency = self.latency_reward(trajectory, time_budget)

        total = (
            self.weights["accuracy"] * acc
            + self.weights["cost"] * cost
            + self.weights["latency"] * latency
        )

        return {
            "total": round(total, 4),
            "accuracy": round(acc, 4),
            "cost": round(cost, 4),
            "latency": round(latency, 4),
        }
