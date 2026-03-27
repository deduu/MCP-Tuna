"""MCP server for data normalization tools."""

import json
from typing import Dict, List

from agentsoul.server import MCPServer
from ..services.normalization_service import DataNormalizationService
from shared.config import NormalizationConfig


class NormalizationMCPServer:
    """Exposes normalization operations as MCP tools."""

    def __init__(self):
        self.service = DataNormalizationService()
        self.mcp = MCPServer("mcp-tuna-normalizer", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(name="normalize.dataset",
                       description="Apply all normalization steps (strip, merge, standardize keys)")
        async def normalize_dataset(
            data_points: List[Dict],
            target_format: str = "sft",
            merge_instruction_input: bool = True,
            strip_whitespace: bool = True,
        ) -> str:
            config = NormalizationConfig(
                target_format=target_format,
                merge_instruction_input=merge_instruction_input,
                strip_whitespace=strip_whitespace,
            )
            result = await svc.normalize_dataset(data_points, config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="normalize.merge_fields",
                       description="Combine instruction + input into single instruction field")
        async def merge_fields(data_points: List[Dict]) -> str:
            result = await svc.merge_instruction_input(data_points)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="normalize.standardize_keys",
                       description="Rename keys to match target format (sft/dpo/grpo)")
        async def standardize_keys(data_points: List[Dict], target_format: str = "sft") -> str:
            result = await svc.standardize_keys(data_points, target_format)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="normalize.remap_fields",
            description="Convert common chat or QA schemas into training-ready rows using a preset",
        )
        async def remap_fields(
            data_points: List[Dict],
            preset: str = "chat_triplet_to_sft",
            keep_unmapped_fields: bool = False,
            strip_whitespace: bool = True,
        ) -> str:
            result = await svc.remap_fields(
                data_points,
                preset=preset,
                keep_unmapped_fields=keep_unmapped_fields,
                strip_whitespace=strip_whitespace,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="normalize.strip_text",
                       description="Strip whitespace and normalize unicode in text fields")
        async def strip_text(data_points: List[Dict]) -> str:
            result = await svc.strip_and_clean_text(data_points)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
