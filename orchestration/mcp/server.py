"""MCP server for orchestration — agent trajectory training data generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentsoul.server import MCPServer
from shared.config import OrchestrationConfig


class OrchestrationMCPServer:
    """Exposes orchestration tools for generating training data from agent trajectories."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._svc = None
        self._config = config or {}
        self.mcp = MCPServer("transcendence-orchestrate", "1.0.0")
        self._register_tools()

    @property
    def service(self):
        if self._svc is None:
            from orchestration.orchestration_trainer import OrchestrationDataService
            from orchestration.rewards import OrchestrationRewardFunction
            from shared.provider_factory import create_llm
            orch_config = OrchestrationConfig(**{
                k: v for k, v in self._config.get("orchestration", {}).items()
                if k in OrchestrationConfig.model_fields
            })
            llm = create_llm(orch_config)
            reward_fn = OrchestrationRewardFunction(
                llm, weights=orch_config.reward_weights,
            )
            self._svc = OrchestrationDataService(llm, reward_fn)
        return self._svc

    def _register_tools(self):
        @self.mcp.tool(
            name="orchestration.generate_problems",
            description="Generate synthetic orchestration tasks for a domain.",
        )
        async def generate_problems(
            domain_description: str,
            num_problems: int = 50,
            tool_descriptions: Optional[List[Dict]] = None,
        ) -> str:
            result = await self.service.generate_problems(
                domain_description, num_problems, tool_descriptions,
            )
            return json.dumps({"success": True, "problems": result, "count": len(result)}, indent=2)

        @self.mcp.tool(
            name="orchestration.collect_trajectories",
            description="Run problems through an agent, record trajectories with cost/latency.",
        )
        async def collect_trajectories(
            problems: List[Dict],
            n_per_problem: int = 4,
        ) -> str:
            from agentsoul.core.agent import AgentSoul
            from shared.provider_factory import create_llm
            from shared.config import PipelineConfig
            agent = AgentSoul(llm_provider=create_llm(PipelineConfig()))
            result = await self.service.collect_trajectories(
                problems, agent, n_per_problem,
            )
            return json.dumps({"success": True, "collected": result, "count": len(result)}, indent=2)

        @self.mcp.tool(
            name="orchestration.build_training_data",
            description="Score trajectories and convert to SFT/DPO/GRPO training format.",
        )
        async def build_training_data(
            collected: List[Dict],
            format: str = "sft",
            tool_descriptions: Optional[List[Dict]] = None,
            cost_budget: float = 1.0,
            time_budget: float = 60.0,
        ) -> str:
            result = await self.service.build_training_data(
                collected, format, tool_descriptions, cost_budget, time_budget,
            )
            return json.dumps({"success": True, "data_points": result, "count": len(result)}, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
