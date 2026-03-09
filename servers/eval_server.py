"""Data evaluation MCP server wrapper.

Re-exports the existing EvaluatorMCPServer with lazy initialization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentsoul.server import MCPServer
from shared.config import EvaluatorConfig


class EvalServer:
    """Thin wrapper around EvaluatorMCPServer with lazy service init."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._inner = None
        self.mcp = MCPServer("mcp-tuna-eval", "1.0.0")
        self._register_tools()

    @property
    def _evaluator(self):
        if self._inner is None:
            from data_evaluator_pipeline.mcp.server import EvaluatorMCPServer
            eval_config = EvaluatorConfig(**self._config.get("evaluator", {}))
            self._inner = EvaluatorMCPServer(eval_config)
        return self._inner

    def _register_tools(self):
        import json
        from typing import List

        @self.mcp.tool(name="evaluate.dataset",
                       description="Score all data points using complexity, IFD, and quality metrics")
        async def evaluate_dataset(
            data_points: List[Dict],
            metrics: Optional[List[str]] = None,
        ) -> str:
            result = await self._evaluator.service.evaluate_dataset(data_points, metrics=metrics)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="evaluate.filter_by_quality",
                       description="Return only entries above a quality threshold")
        async def filter_by_quality(
            data_points: List[Dict],
            threshold: float = 0.7,
            metrics: Optional[List[str]] = None,
        ) -> str:
            result = await self._evaluator.service.filter_by_quality(data_points, threshold, metrics=metrics)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="evaluate.statistics",
                       description="Return per-metric min/max/mean/stdev statistics")
        async def analyze_statistics(
            data_points: List[Dict],
            metrics: Optional[List[str]] = None,
        ) -> str:
            result = await self._evaluator.service.analyze_statistics(data_points, metrics=metrics)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="evaluate.list_metrics",
                       description="List all registered evaluation metrics")
        async def list_metrics() -> str:
            return json.dumps(self._evaluator.service.list_metrics(), indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
