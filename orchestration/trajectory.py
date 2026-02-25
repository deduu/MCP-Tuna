"""
Trajectory recorder for AgentSoul runs.

Wraps agent.run() and captures structured trajectories with
cost/latency metadata — zero changes to agent.py.

The recorder consumes events from agent.run(). In streaming mode the
externally-visible events are: token, tool_exec_start, tool_exec_end,
phase_start, phase_end, complete.  In non-streaming mode only a single
'complete' event is yielded.  The 'complete' event carries the full
agent history and usage dict which the recorder uses to build the
structured trajectory.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TurnRecord:
    turn_number: int
    model_id: str
    tool_calls: List[Dict] = field(default_factory=list)
    llm_tokens: Dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    turn_latency_s: float = 0.0
    estimated_cost_usd: float = 0.0


@dataclass
class Trajectory:
    task: str
    turns: List[TurnRecord] = field(default_factory=list)
    final_answer: str = ""
    total_latency_s: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    model_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrajectoryRecorder:
    """Record structured trajectories from AgentSoul runs."""

    COST_TABLE: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 2.50 / 1e6, "output": 10.0 / 1e6},
        "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
        "o3-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
        "local": {"input": 0.0, "output": 0.0},
    }

    def __init__(self, cost_table: Optional[Dict[str, Dict[str, float]]] = None):
        if cost_table:
            self.COST_TABLE.update(cost_table)

    def _get_cost_rates(self, model_id: str) -> Dict[str, float]:
        """Look up per-token cost rates, falling back to local (free)."""
        return self.COST_TABLE.get(model_id, self.COST_TABLE["local"])

    def _estimate_cost(self, model_id: str, usage: Dict[str, int]) -> float:
        rates = self._get_cost_rates(model_id)
        prompt_cost = usage.get("prompt_tokens", 0) * rates["input"]
        completion_cost = usage.get("completion_tokens", 0) * rates["output"]
        return prompt_cost + completion_cost

    async def record(self, agent, user_input: str, **kwargs) -> Trajectory:
        """
        Run agent.run() and capture the full trajectory.

        Consumes the async event stream without modifying the agent.
        Uses tool_exec_start/end for per-tool latency and the 'complete'
        event's history for turn structure and tool call details.
        """
        model_id = getattr(agent, "model_id", "unknown")
        trajectory = Trajectory(task=user_input, model_id=model_id)

        global_start = time.perf_counter()

        # Track tool latencies as they happen via events
        tool_latencies: Dict[str, List[float]] = {}  # tool_name -> [latency, ...]
        active_tool_start: Optional[float] = None
        active_tool_name: Optional[str] = None

        async for event in agent.run(user_input, **kwargs):
            etype = event.get("type")

            if etype == "tool_exec_start":
                active_tool_start = time.perf_counter()
                active_tool_name = event.get("tool", "")

            elif etype == "tool_exec_end":
                tool_latency = (
                    time.perf_counter() - active_tool_start
                    if active_tool_start
                    else 0.0
                )
                tool_name = event.get("tool", active_tool_name or "")
                tool_latencies.setdefault(tool_name, []).append(
                    round(tool_latency, 4)
                )
                active_tool_start = None
                active_tool_name = None

            elif etype == "complete":
                trajectory.final_answer = event.get("content", "")
                usage = event.get("usage") or {}
                history = event.get("history", [])

                # Build TurnRecords from agent history
                # Track which tool latency we've consumed per tool name
                latency_idx: Dict[str, int] = {}

                for hist_turn in history:
                    turn = TurnRecord(
                        turn_number=hist_turn.get("turn_number", 0),
                        model_id=model_id,
                    )

                    for tc in hist_turn.get("tool_calls", []):
                        name = tc.get("name", "")
                        # Pop next latency measurement for this tool
                        idx = latency_idx.get(name, 0)
                        lats = tool_latencies.get(name, [])
                        lat = lats[idx] if idx < len(lats) else 0.0
                        latency_idx[name] = idx + 1

                        turn.tool_calls.append({
                            "name": name,
                            "arguments": tc.get("arguments", {}),
                            "result": tc.get("result", ""),
                            "latency_s": lat,
                        })

                    trajectory.turns.append(turn)

                # Distribute usage across turns (approximate: split evenly)
                if usage and trajectory.turns:
                    per_turn_prompt = usage.get("prompt_tokens", 0) // len(trajectory.turns)
                    per_turn_completion = usage.get("completion_tokens", 0) // len(trajectory.turns)
                    for turn in trajectory.turns:
                        turn.llm_tokens = {
                            "prompt_tokens": per_turn_prompt,
                            "completion_tokens": per_turn_completion,
                        }
                        turn.estimated_cost_usd = self._estimate_cost(
                            model_id, turn.llm_tokens
                        )

                trajectory.total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        trajectory.total_latency_s = round(
            time.perf_counter() - global_start, 4
        )
        trajectory.total_cost_usd = round(
            sum(t.estimated_cost_usd for t in trajectory.turns), 6
        )
        return trajectory
