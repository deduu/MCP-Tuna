"""MCP server for data evaluation tools."""

import json
from typing import Any, Dict, List, Optional

from agentsoul.server import MCPServer
from ..services.pipeline_service import EvaluatorService
from shared.config import EvaluatorConfig


class EvaluatorMCPServer:
    """Exposes evaluation operations as MCP tools."""

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        self.service = EvaluatorService(config)
        self.mcp = MCPServer("evaluator-pipeline", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(name="evaluate.dataset",
                       description="Score all data points using complexity, IFD, and quality metrics")
        async def evaluate_dataset(data_points: List[Dict]) -> str:
            result = await svc.evaluate_dataset(data_points)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="evaluate.filter_by_quality",
                       description="Return only entries above a quality threshold")
        async def filter_by_quality(
            data_points: List[Dict],
            threshold: float = 0.7,
        ) -> str:
            result = await svc.filter_by_quality(data_points, threshold)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="evaluate.statistics",
                       description="Return per-metric min/max/mean/stdev statistics")
        async def analyze_statistics(data_points: List[Dict]) -> str:
            result = await svc.analyze_statistics(data_points)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="evaluate.list_metrics",
                       description="List all registered evaluation metrics")
        async def list_metrics() -> str:
            return json.dumps(svc.list_metrics(), indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
