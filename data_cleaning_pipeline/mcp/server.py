"""MCP server for data cleaning tools."""

import json
from typing import Any, Dict, List, Optional

from agentsoul.server import MCPServer
from ..services.cleaning_service import DataCleaningService
from shared.config import CleaningConfig


class CleaningMCPServer:
    """Exposes cleaning operations as MCP tools."""

    def __init__(self):
        self.service = DataCleaningService()
        self.mcp = MCPServer("cleaning-pipeline", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(name="clean.dataset",
                       description="Run all cleaning steps (deduplicate, validate, filter short entries)")
        async def clean_dataset(
            data_points: List[Dict],
            remove_duplicates: bool = True,
            min_instruction_length: int = 10,
            min_output_length: int = 20,
        ) -> str:
            config = CleaningConfig(
                remove_duplicates=remove_duplicates,
                min_instruction_length=min_instruction_length,
                min_output_length=min_output_length,
            )
            result = await svc.clean_dataset(data_points, config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="clean.deduplicate",
                       description="Remove exact duplicate entries by key")
        async def deduplicate(data_points: List[Dict], key: str = "instruction") -> str:
            result = await svc.deduplicate(data_points, key)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="clean.validate_schema",
                       description="Validate entries have required fields for a technique (sft/dpo/grpo)")
        async def validate_schema(data_points: List[Dict], technique: str = "sft") -> str:
            result = await svc.validate_schema(data_points, technique)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="clean.remove_short",
                       description="Filter entries below minimum length thresholds")
        async def remove_short(
            data_points: List[Dict],
            min_instruction: int = 10,
            min_output: int = 20,
        ) -> str:
            result = await svc.remove_short_entries(data_points, min_instruction, min_output)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
