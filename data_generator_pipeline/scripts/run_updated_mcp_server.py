"""
Fine-tuning Pipeline MCP Server
================================

Exposes the fine-tuning pipeline as MCP tools for AI agents.
"""

from ..services.pipeline_service import PipelineService
from agentsoul.server import MCPServer, StdioTransport, HTTPTransport
from typing import Optional
import json
import sys
import os
from dotenv import load_dotenv
from agentsoul.providers.openai import OpenAIProvider

load_dotenv(override=True)  # override system env if needed
model = os.getenv("OPENAI_MODEL", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

# Import your MCP server builder
# Import your pipeline service


class FineTuningMCPServer:
    """MCP Server wrapper for fine-tuning pipeline."""

    def __init__(self, llm_provider, config: dict):
        """Initialize the MCP server with pipeline service."""
        self.mcp = MCPServer(
            name="finetuning-pipeline",
            version="1.0.0"
        )

        # Initialize the pipeline service
        self.service = PipelineService(llm_provider, config)

        # Register all tools
        self._register_tools()

    def _register_tools(self):
        """Register all pipeline operations as MCP tools."""

        @self.mcp.tool(
            name="load_document",
            description="Load and parse a document (PDF, DOCX, etc.) for processing"
        )
        async def load_document(file_path: str) -> str:
            """Load a document and return its structure."""
            result = await self.service.load_document(file_path)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="generate_from_page",
            description="Generate fine-tuning data from a single page of a document"
        )
        async def generate_from_page(
            technique: str,
            page_text: str,
            page_index: int,
            file_name: str,
            custom_template: Optional[str] = None
        ) -> str:
            """Generate training data from a single page."""
            result = await self.service.generate_from_page(
                technique=technique,
                page_text=page_text,
                page_index=page_index,
                file_name=file_name,
                custom_template=custom_template
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="generate_from_document",
            description="Generate fine-tuning data from an entire document"
        )
        async def generate_from_document(
            technique: str,
            file_path: str,
            custom_template: Optional[str] = None,
            start_page: Optional[int] = None,
            end_page: Optional[int] = None
        ) -> str:
            """Generate training data from a full document."""
            result = await self.service.generate_from_document(
                technique=technique,
                file_path=file_path,
                custom_template=custom_template,
                start_page=start_page,
                end_page=end_page
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="generate_batch",
            description="Generate fine-tuning data from multiple documents"
        )
        async def generate_batch(
            technique: str,
            file_paths: str,  # JSON string of file paths
            custom_template: Optional[str] = None
        ) -> str:
            """Generate training data from multiple documents."""
            # Parse file_paths from JSON string
            paths = json.loads(file_paths) if isinstance(
                file_paths, str) else file_paths

            result = await self.service.generate_batch(
                technique=technique,
                file_paths=paths,
                custom_template=custom_template
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="export_dataset",
            description="Export generated dataset to a file (JSON, JSONL, CSV, Excel, or HuggingFace)"
        )
        async def export_dataset(
            data_points: str,  # JSON string of data points
            output_path: str,
            format: str = "jsonl"
        ) -> str:
            """Export dataset to file."""
            # Parse data_points from JSON string
            points = json.loads(data_points) if isinstance(
                data_points, str) else data_points

            result = await self.service.export_dataset(
                data_points=points,
                output_path=output_path,
                format=format
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="list_techniques",
            description="List all available fine-tuning techniques"
        )
        def list_techniques() -> str:
            """List available fine-tuning techniques."""
            result = self.service.list_techniques()
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="get_technique_schema",
            description="Get the data schema for a specific fine-tuning technique"
        )
        def get_technique_schema(technique: str) -> str:
            """Get schema for a technique."""
            result = self.service.get_technique_schema(technique)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        """Start the MCP server."""
        self.mcp.run(transport)


# Example usage and startup script
if __name__ == "__main__":
    """
    Usage:
        # Run with stdio (for Claude Desktop, etc.)
        python finetuning_mcp_server.py

        # Run with HTTP server
        python finetuning_mcp_server.py http
        python finetuning_mcp_server.py http 8080  # custom port
    """

    llm_provider = OpenAIProvider(
        model_id=model, api_key=api_key, base_url=base_url)

    # Pipeline configuration
    config = {
        "debug": False,
        "max_workers": 4,
        "batch_size": 10,
    }

    # ===================================================================
    # SERVER INITIALIZATION
    # ===================================================================

    # Create the MCP server
    server = FineTuningMCPServer(llm_provider, config)

    # Determine transport based on arguments
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        # HTTP mode - for testing with curl or web clients
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        transport = HTTPTransport(host="0.0.0.0", port=port)
    else:
        # Stdio mode - for Claude Desktop and MCP-compatible clients
        transport = StdioTransport()

    # Start the server
    print("Starting Fine-tuning Pipeline MCP Server...", file=sys.stderr)
    print("Available techniques will be listed on first connection", file=sys.stderr)
    server.run(transport)
