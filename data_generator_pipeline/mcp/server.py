"""MCP Server for the Data Generator Pipeline.

Uses agentsoul's MCPServer framework with auto-schema
so tools are derived from type hints — no manual inputSchema dicts.
"""

from typing import Any, Dict, List, Optional, Union
import json

from agentsoul.server import MCPServer
from shared.config import GeneratorConfig
from ..services.pipeline_service import PipelineService


class GeneratorMCPServer:
    """Exposes all generator pipeline operations as MCP tools."""

    def __init__(self, llm_provider, config: Union[GeneratorConfig, Dict[str, Any]]):
        self.service = PipelineService(llm_provider, config)
        self.mcp = MCPServer("generator-pipeline", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(name="generate.load_document",
                       description="Load and parse a document file (PDF, Markdown, etc.)")
        async def load_document(file_path: str) -> str:
            result = await svc.load_document(file_path)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="generate.from_page",
                       description="Generate fine-tuning data from a single page")
        async def generate_from_page(
            technique: str,
            page_text: str,
            page_index: int,
            file_name: str,
            custom_template: Optional[str] = None,
        ) -> str:
            result = await svc.generate_from_page(
                technique=technique,
                page_text=page_text,
                page_index=page_index,
                file_name=file_name,
                custom_template=custom_template,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="generate.from_document",
                       description="Generate fine-tuning data from an entire document")
        async def generate_from_document(
            technique: str,
            file_path: str,
            custom_template: Optional[str] = None,
            start_page: Optional[int] = None,
            end_page: Optional[int] = None,
        ) -> str:
            result = await svc.generate_from_document(
                technique=technique,
                file_path=file_path,
                custom_template=custom_template,
                start_page=start_page,
                end_page=end_page,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="generate.batch",
                       description="Generate fine-tuning data from multiple documents")
        async def generate_batch(
            technique: str,
            file_paths: List[str],
            custom_template: Optional[str] = None,
        ) -> str:
            result = await svc.generate_batch(
                technique=technique,
                file_paths=file_paths,
                custom_template=custom_template,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="generate.export_dataset",
                       description="Export generated dataset to file")
        async def export_dataset(
            data_points: List[Dict],
            output_path: str,
            format: str = "jsonl",
        ) -> str:
            result = await svc.export_dataset(data_points, output_path, format)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="generate.list_techniques",
                       description="List all available fine-tuning techniques")
        async def list_techniques() -> str:
            return json.dumps(svc.list_techniques(), indent=2)

        @self.mcp.tool(name="generate.get_schema",
                       description="Get the data schema for a specific technique")
        async def get_technique_schema(technique: str) -> str:
            return json.dumps(svc.get_technique_schema(technique), indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
