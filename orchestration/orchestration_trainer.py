"""
End-to-end service for generating orchestration training data.

Generates synthetic problems, collects agent trajectories at varying
temperatures, scores them, and converts to SFT/DPO/GRPO training format.

Training data is **schema-aware**: tool descriptions are embedded in the
instruction context so the fine-tuned model learns routing *patterns*,
not specific tool names.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from AgentY.data_generator_pipeline.parsers.json_extractor import JsonExtractor
from AgentY.data_generator_pipeline.exporters.dataset import DatasetExporter
from AgentY.orchestration.trajectory import Trajectory, TrajectoryRecorder
from AgentY.orchestration.rewards import OrchestrationRewardFunction


PROBLEM_GEN_PROMPT = """You are a task designer for AI agent evaluation.

Domain: {domain}

Available tools:
{tool_descriptions}

Generate exactly {num_problems} diverse tasks that require multi-step tool use in this domain.
Each task should vary in difficulty and require different tool combinations.

Return a JSON array where each element has:
- "task": the user-facing task description
- "expected_tools": list of tool names likely needed
- "difficulty": "easy" | "medium" | "hard"
- "ground_truth_hint": a brief description of what a correct answer should contain

Return ONLY the JSON array, no other text."""


def _format_tool_descriptions(tool_descriptions: Optional[List[Dict]]) -> str:
    """Format tool descriptions for embedding in prompts."""
    if not tool_descriptions:
        return "(no tools provided)"
    lines = []
    for tool in tool_descriptions:
        name = tool.get("name", tool.get("function", {}).get("name", "unknown"))
        desc = tool.get("description", tool.get("function", {}).get("description", ""))
        params = tool.get("parameters", tool.get("function", {}).get("parameters", {}))
        lines.append(f"- {name}: {desc}\n  Parameters: {json.dumps(params)}")
    return "\n".join(lines)


class OrchestrationDataService:
    """Generate orchestration training data from agent trajectories."""

    def __init__(
        self,
        llm,
        reward_fn: OrchestrationRewardFunction,
        recorder: Optional[TrajectoryRecorder] = None,
    ):
        self.llm = llm
        self.reward_fn = reward_fn
        self.recorder = recorder or TrajectoryRecorder()
        self.parser = JsonExtractor()

    async def generate_problems(
        self,
        domain_description: str,
        num_problems: int = 50,
        tool_descriptions: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Use LLM to generate synthetic tasks requiring tool orchestration."""
        prompt = PROBLEM_GEN_PROMPT.format(
            domain=domain_description,
            tool_descriptions=_format_tool_descriptions(tool_descriptions),
            num_problems=num_problems,
        )
        response = await self.llm.chat(
            [{"role": "user", "content": prompt}], tools=None
        )
        return self.parser.extract(response.content)

    async def collect_trajectories(
        self,
        problems: List[Dict],
        agent,
        n_per_problem: int = 4,
        temperatures: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        For each problem, run the agent N times at different temperatures.
        Record trajectories via TrajectoryRecorder.
        """
        temperatures = temperatures or [0.3, 0.5, 0.7, 1.0]
        # Pad or trim temperatures to match n_per_problem
        temps = (temperatures * ((n_per_problem // len(temperatures)) + 1))[
            :n_per_problem
        ]

        collected = []
        for problem in problems:
            task = problem["task"]
            trajectories: List[Trajectory] = []

            for temp in temps:
                trajectory = await self.recorder.record(
                    agent, task, temperature=temp
                )
                trajectories.append(trajectory)

            collected.append({
                "problem": problem,
                "trajectories": [t.to_dict() for t in trajectories],
            })

        return collected

    async def build_training_data(
        self,
        collected: List[Dict],
        format: str = "sft",
        tool_descriptions: Optional[List[Dict]] = None,
        cost_budget: float = 1.0,
        time_budget: float = 60.0,
    ) -> List[Dict]:
        """
        Score trajectories and convert to training format.

        Schema-aware: tool descriptions are embedded in the instruction
        context so the model learns routing patterns, not tool names.
        """
        tool_context = _format_tool_descriptions(tool_descriptions)

        scored_groups: List[Dict] = []
        for item in collected:
            problem = item["problem"]
            rewards = []

            for traj_dict in item["trajectories"]:
                traj = self._dict_to_trajectory(traj_dict)
                reward = await self.reward_fn.compute(
                    traj,
                    ground_truth=problem.get("ground_truth_hint"),
                    budget=cost_budget,
                    time_budget=time_budget,
                )
                rewards.append(reward)

            scored_groups.append({
                "problem": problem,
                "trajectories": item["trajectories"],
                "rewards": rewards,
            })

        if format == "sft":
            return self._to_sft(scored_groups, tool_context)
        elif format == "dpo":
            return self._to_dpo(scored_groups, tool_context)
        elif format == "grpo":
            return self._to_grpo(scored_groups, tool_context)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'sft', 'dpo', or 'grpo'.")

    def _build_instruction(self, task: str, tool_context: str) -> str:
        """Build schema-aware instruction with tool descriptions."""
        return (
            f"You are an orchestration agent. Given the user task and available tools, "
            f"determine the optimal sequence of tool calls to complete the task "
            f"efficiently (minimizing cost and latency while maximizing accuracy).\n\n"
            f"Available tools:\n{tool_context}\n\n"
            f"Task: {task}"
        )

    def _serialize_trajectory(self, traj_dict: Dict) -> str:
        """Serialize a trajectory into a readable routing plan."""
        steps = []
        for turn in traj_dict.get("turns", []):
            for tc in turn.get("tool_calls", []):
                args = tc.get("arguments", {})
                steps.append(f"Step {len(steps)+1}: {tc['name']}({json.dumps(args)})")
        if not steps:
            steps.append("(no tool calls — direct answer)")
        steps.append(f"Final answer: {traj_dict.get('final_answer', '')[:500]}")
        return "\n".join(steps)

    def _to_sft(self, scored_groups: List[Dict], tool_context: str) -> List[Dict]:
        """Select best trajectory per problem."""
        data = []
        for group in scored_groups:
            rewards = group["rewards"]
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
            best_traj = group["trajectories"][best_idx]

            data.append({
                "instruction": self._build_instruction(
                    group["problem"]["task"], tool_context
                ),
                "output": self._serialize_trajectory(best_traj),
            })
        return data

    def _to_dpo(self, scored_groups: List[Dict], tool_context: str) -> List[Dict]:
        """Pair best vs worst trajectory."""
        data = []
        for group in scored_groups:
            rewards = group["rewards"]
            if len(rewards) < 2:
                continue
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
            worst_idx = min(range(len(rewards)), key=lambda i: rewards[i]["total"])
            if best_idx == worst_idx:
                continue

            data.append({
                "prompt": self._build_instruction(
                    group["problem"]["task"], tool_context
                ),
                "chosen": self._serialize_trajectory(
                    group["trajectories"][best_idx]
                ),
                "rejected": self._serialize_trajectory(
                    group["trajectories"][worst_idx]
                ),
            })
        return data

    def _to_grpo(self, scored_groups: List[Dict], tool_context: str) -> List[Dict]:
        """Keep all trajectories with their reward scores."""
        data = []
        for group in scored_groups:
            data.append({
                "prompt": self._build_instruction(
                    group["problem"]["task"], tool_context
                ),
                "responses": [
                    self._serialize_trajectory(t) for t in group["trajectories"]
                ],
                "rewards": [r["total"] for r in group["rewards"]],
            })
        return data

    @staticmethod
    def _dict_to_trajectory(d: Dict) -> Trajectory:
        """Reconstruct a Trajectory from its dict representation."""
        return Trajectory(
            task=d.get("task", ""),
            final_answer=d.get("final_answer", ""),
            total_latency_s=d.get("total_latency_s", 0.0),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            total_tokens=d.get("total_tokens", 0),
            model_id=d.get("model_id", ""),
        )

    async def export(
        self,
        data: List[Dict],
        output_path: str,
        file_format: str = "jsonl",
    ) -> Dict[str, Any]:
        """Export training data to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if file_format == "jsonl":
            DatasetExporter.to_jsonl(data, output_path)
        elif file_format == "json":
            DatasetExporter.to_json(data, output_path)
        elif file_format == "huggingface":
            DatasetExporter.to_huggingface(data, output_path)
        else:
            DatasetExporter.to_jsonl(data, output_path)

        return {
            "success": True,
            "output_path": output_path,
            "format": file_format,
            "num_examples": len(data),
        }
