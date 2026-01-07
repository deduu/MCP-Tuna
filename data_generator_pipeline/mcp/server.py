# ============================================================================
# FILE: src/finetuning/mcp/server.py
# ============================================================================

"""
MCP Server for Fine-tuning Pipeline.

This exposes all pipeline operations as MCP tools that AI agents can call.
"""

from typing import Any, Dict
import asyncio
import sys
from mcp.server import Server
from mcp.types import Tool, TextContent

from ..services.pipeline_service import PipelineService


class FineTuningMCPServer:
    """MCP Server for fine-tuning pipeline."""

    def __init__(self, llm_provider, config: Dict[str, Any]):
        self.service = PipelineService(llm_provider, config)
        self.server = Server("finetuning-pipeline")
        self._register_tools()

    def _register_tools(self):
        """Register all tools with the MCP server."""

        # Tool 1: Load Document
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="load_document",
                    description="Load and parse a document file (PDF, Markdown, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the document file"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="generate_from_page",
                    description="Generate fine-tuning data from a single page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "technique": {
                                "type": "string",
                                "enum": ["sft", "dpo", "grpo"],
                                "description": "Fine-tuning technique"
                            },
                            "page_text": {
                                "type": "string",
                                "description": "Text content of the page"
                            },
                            "page_index": {
                                "type": "integer",
                                "description": "Index of the page"
                            },
                            "file_name": {
                                "type": "string",
                                "description": "Name of the source file"
                            },
                            "custom_template": {
                                "type": "string",
                                "description": "Optional custom prompt template"
                            }
                        },
                        "required": ["technique", "page_text", "page_index", "file_name"]
                    }
                ),
                Tool(
                    name="generate_from_document",
                    description="Generate fine-tuning data from an entire document",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "technique": {
                                "type": "string",
                                "enum": ["sft", "dpo", "grpo"],
                                "description": "Fine-tuning technique"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to the document file"
                            },
                            "custom_template": {
                                "type": "string",
                                "description": "Optional custom prompt template"
                            },
                            "start_page": {
                                "type": "integer",
                                "description": "Start page (optional)"
                            },
                            "end_page": {
                                "type": "integer",
                                "description": "End page (optional)"
                            }
                        },
                        "required": ["technique", "file_path"]
                    }
                ),
                Tool(
                    name="generate_batch",
                    description="Generate fine-tuning data from multiple documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "technique": {
                                "type": "string",
                                "enum": ["sft", "dpo", "grpo"],
                                "description": "Fine-tuning technique"
                            },
                            "file_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of document file paths"
                            },
                            "custom_template": {
                                "type": "string",
                                "description": "Optional custom prompt template"
                            }
                        },
                        "required": ["technique", "file_paths"]
                    }
                ),
                Tool(
                    name="export_dataset",
                    description="Export generated dataset to file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data_points": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "Array of data points to export"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path for output file"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["json", "jsonl", "excel", "csv", "huggingface"],
                                "description": "Output format"
                            }
                        },
                        "required": ["data_points", "output_path", "format"]
                    }
                ),
                Tool(
                    name="list_techniques",
                    description="List all available fine-tuning techniques",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_technique_schema",
                    description="Get the data schema for a specific technique",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "technique": {
                                "type": "string",
                                "description": "Technique name"
                            }
                        },
                        "required": ["technique"]
                    }
                ),
            ]

        # Tool handlers
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            print(
                f"Tool called: {name} with args: {arguments}", file=sys.stderr)

            if name == "load_document":
                result = await self.service.load_document(**arguments)
            elif name == "generate_from_page":
                result = await self.service.generate_from_page(**arguments)
            elif name == "generate_from_document":
                result = await self.service.generate_from_document(**arguments)
            elif name == "generate_batch":
                result = await self.service.generate_batch(**arguments)
            elif name == "export_dataset":
                result = await self.service.export_dataset(**arguments)
            elif name == "list_techniques":
                result = self.service.list_techniques()
            elif name == "get_technique_schema":
                result = self.service.get_technique_schema(**arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}

            import json
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

    async def run(self):
        """Run the MCP server."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )
