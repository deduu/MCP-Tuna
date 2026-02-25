"""
Fine-tuning MCP Server
======================

Exposes fine-tuning pipeline as MCP tools for AI agents.
"""

import sys
import json
from typing import Optional
from pathlib import Path

# Import MCP server framework
from agentsoul.server import MCPServer, StdioTransport, HTTPTransport

# Import fine-tuning service
from ..services.pipeline_service import FineTuningService


class FineTuningMCP:
    """MCP server for fine-tuning operations."""
    
    def __init__(self, default_base_model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize the MCP server."""
        self.mcp = MCPServer(
            name="finetuning-pipeline",
            version="1.0.0"
        )
        
        # Initialize service
        self.service = FineTuningService(default_base_model=default_base_model)
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all fine-tuning tools."""
        
        # ============================================================
        # DATASET TOOLS
        # ============================================================
        
        @self.mcp.tool(
            name="load_dataset",
            description="Load a dataset from a JSON, JSONL, or CSV file"
        )
        async def load_dataset(
            file_path: str,
            format: str = "json"
        ) -> str:
            """Load dataset from file."""
            result = await self.service.load_dataset_from_file(file_path, format)
            # Remove dataset object from response (not JSON serializable)
            if "dataset_object" in result:
                del result["dataset_object"]
            return json.dumps(result, indent=2)
        
        @self.mcp.tool(
            name="prepare_dataset",
            description="Prepare a dataset from raw data (JSON string of list of dicts)"
        )
        async def prepare_dataset(
            data: str,
            prompt_column: str = "instruction",
            response_column: str = "response"
        ) -> str:
            """Prepare dataset from raw data."""
            # Parse JSON string
            data_list = json.loads(data) if isinstance(data, str) else data
            result = await self.service.prepare_dataset(
                data_list,
                prompt_column=prompt_column,
                response_column=response_column
            )
            # Remove dataset object from response
            if "dataset_object" in result:
                del result["dataset_object"]
            return json.dumps(result, indent=2)
        
        # ============================================================
        # TRAINING TOOLS
        # ============================================================
        
        @self.mcp.tool(
            name="train_model",
            description=(
                "Fine-tune a base language model using a dataset. "
                "Use this tool ONLY when the user explicitly asks to train, "
                "fine-tune, adapt, or specialize a model. "
                "Do NOT use this tool for text generation or inference. "
                "If the base model path is unknown, first call "
                "list_available_base_models."
            )
        )
        async def train_model(
            dataset_path: str,
            output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            prompt_column: str = "prompt",
            response_column: str = "response"
        ) -> str:
            """Train a model."""
            # Load dataset
            load_result = await self.service.load_dataset_from_file(dataset_path, "json")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            
            dataset = load_result["dataset_object"]
            
            # Train
            result = await self.service.train_model(
                dataset=dataset,
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                prompt_column=prompt_column,
                response_column=response_column
            )
            
            return json.dumps(result, indent=2)
        
        @self.mcp.tool(
            name="train_from_data",
            description=(
                "Fine-tune a model using inline training data. "
                "Use this tool only for quick experiments or small datasets "
                "provided directly by the user. "
                "Do NOT use this tool for inference."
            )
        )
        async def train_from_data(
            data: str,
            output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            prompt_column: str = "instruction",
            response_column: str = "response"
        ) -> str:
            """Train model from inline data."""
            # Parse data
            data_list = json.loads(data) if isinstance(data, str) else data
            
            # Prepare dataset
            prep_result = await self.service.prepare_dataset(
                data_list,
                prompt_column=prompt_column,
                response_column=response_column
            )
            
            if not prep_result["success"]:
                return json.dumps(prep_result, indent=2)
            
            dataset = prep_result["dataset_object"]
            
            # Train
            result = await self.service.train_model(
                dataset=dataset,
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                use_lora=use_lora,
                lora_r=lora_r,
                prompt_column="prompt",
                response_column=response_column
            )
            
            return json.dumps(result, indent=2)
        
        # ============================================================
        # INFERENCE TOOLS
        # ============================================================
        
        @self.mcp.tool(
        name="run_inference",
        description=(
            "Generate text using an existing model. "
            "Use this tool when the user asks to generate, answer, test, "
            "or run inference. "
            "This tool MUST NOT be used for training or fine-tuning. "
            "If a model path is not known, resolve it first using "
            "list_available_base_models."
        )
    )
        async def run_inference(
            prompts: str | list,
            model_path: str,
            adapter_path: Optional[str] = None,
            max_new_tokens: int = 512,
            temperature: float = 0.7
        ) -> str:
            """Run inference."""

            # ✅ Normalize prompts (MCP-safe)
            if isinstance(prompts, str):
                prompts_list = [prompts]
            elif isinstance(prompts, list):
                prompts_list = prompts
            else:
                return json.dumps({
                    "success": False,
                    "error": "prompts must be a string or list of strings"
                }, indent=2)

            result = await self.service.run_inference(
                prompts=prompts_list,
                model_path=model_path,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            return json.dumps(result, indent=2)

        
        @self.mcp.tool(
            name="compare_models",
            description=(
                "Compare outputs from a base model and a fine-tuned adapter "
                "using the same prompts. "
                "Use this tool only for evaluation or benchmarking."
            )
        )
        async def compare_models(
            prompts: str,
            base_model_path: str,
            finetuned_adapter_path: str,
            max_new_tokens: int = 512
        ) -> str:
            """Compare models."""
            # Parse prompts
            prompts_list = json.loads(prompts) if isinstance(prompts, str) else prompts
            
            result = await self.service.compare_models(
                prompts=prompts_list,
                base_model_path=base_model_path,
                finetuned_adapter_path=finetuned_adapter_path,
                max_new_tokens=max_new_tokens
            )
            
            return json.dumps(result, indent=2)
        
        @self.mcp.tool(
        name="list_available_base_models",
        description=(
            "Discover which base language models are already available locally "
            "on this machine. "
            "Use this tool ONLY when you need a valid local model path for "
            "training or inference. "
            "Do NOT call this tool unless a model path is required."
            )
        )
        async def list_available_base_models(
            query: str = ""
        ) -> str:
            """
            Resolve locally available base model paths for AI agents.
            """
            result = await self.service.list_available_base_models(query=query)
            return json.dumps(result, indent=2)


        @self.mcp.tool(
            name="search_huggingface_models",
            description=(
                "Search for available models on the Hugging Face Hub. "
                "Use this tool ONLY when a suitable local model is not available "
                "and you need to explore remote base models. "
                "This tool does NOT download models."
            )
        )
        async def search_huggingface_models(
            query: str = "",
            task: str = "text-generation",
            sort: str = "downloads",
            limit: int = 20
        ) -> str:
            """
            Search remote Hugging Face models.
            """
            result = await self.service.search_huggingface_models(
                query=query,
                task=task,
                sort=sort,
                limit=limit
            )
            return json.dumps(result, indent=2)
        

        @self.mcp.tool(
            name="get_huggingface_model_info",
            description=(
                "Retrieve detailed metadata about a specific Hugging Face model. "
                "Use this tool ONLY after a model has been identified "
                "(e.g., from search_huggingface_models). "
                "Do NOT use this tool to select a model."
            )
        )
        async def get_huggingface_model_info(
            model_id: str
        ) -> str:
            """
            Inspect a specific Hugging Face model.
            """
            result = await self.service.get_huggingface_model_info(model_id)
            return json.dumps(result, indent=2)


        # ============================================================
        # MODEL MANAGEMENT TOOLS
        # ============================================================
        
        @self.mcp.tool(
            name="get_model_info",
            description="Get information about a model or adapter"
        )
        def get_model_info(model_path: str) -> str:
            """Get model info."""
            result = self.service.get_model_info(model_path)
            return json.dumps(result, indent=2)
        
        @self.mcp.tool(
            name="search_local_models",
            description="Search locally cached HuggingFace models with file details"
        )
        async def search_local_models(query: str = "") -> str:
            """Search local model cache."""
            result = await self.service.search_local_models(query=query)
            return json.dumps(result, indent=2)
        
        @self.mcp.tool(
            name="clear_gpu_memory",
            description="Clear GPU memory cache"
        )
        def clear_gpu_memory() -> str:
            """Clear GPU memory."""
            result = self.service.clear_gpu_memory()
            return json.dumps(result, indent=2)
    
    def run(self, transport=None):
        """Start the MCP server."""
        self.mcp.run(transport)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Usage:
        # Stdio mode (for Claude Desktop)
        python finetuning_mcp.py
        
        # HTTP mode (for testing)
        python finetuning_mcp.py http 8000
    """
    
    # Configuration
    DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Create server
    server = FineTuningMCP(default_base_model=DEFAULT_BASE_MODEL)
    
    # Determine transport
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        transport = HTTPTransport(host="0.0.0.0", port=port)
        print(f"\n🚀 Starting Fine-tuning MCP Server on HTTP port {port}")
        print(f"   Test with: curl http://localhost:{port}/health\n")
    else:
        transport = StdioTransport()
        print("🚀 Starting Fine-tuning MCP Server (stdio mode)", file=sys.stderr)
    
    # Start server
    server.run(transport)