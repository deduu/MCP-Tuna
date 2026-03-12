"""
MCP-Ready Fine-tuning Pipeline Architecture

This refactoring makes the pipeline easily exposable as MCP tools while
maintaining backward compatibility with the current CLI/API usage.

Key changes:
1. Service layer that wraps pipeline operations
2. Stateless operations with clear inputs/outputs
3. JSON serializable results
4. MCP server implementation
5. Tool schemas for AI agent discovery
"""

# ============================================================================
# FILE: src/finetuning/services/pipeline_service.py
# ============================================================================

from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import asdict

from ..core.factory import PipelineFactory
from ..parsers.json_extractor import JsonExtractor
from ..prompts.templates import PromptTemplateManager
from ..exporters.dataset import DatasetExporter
from ..loaders import get_loader
from shared.config import GeneratorConfig


class PipelineService:
    """
    Service layer for fine-tuning pipeline operations.
    Stateless methods with JSON-serializable inputs/outputs.
    Perfect for MCP tool exposure.
    """

    def __init__(self, llm_provider, config: Union[GeneratorConfig, Dict[str, Any]]):
        """Initialize service with LLM provider and config."""
        self.llm_provider = llm_provider
        if isinstance(config, GeneratorConfig):
            self.config = config.model_dump()
            self.generator_config = config
        else:
            self.config = config
            self.generator_config = GeneratorConfig(**{
                k: v for k, v in config.items()
                if k in GeneratorConfig.model_fields
            })
        self.template_manager = PromptTemplateManager()
        self.parser = JsonExtractor()

    async def load_document(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Load and parse a document.

        MCP Tool: load_document
        Returns: {"file_name": str, "pages": [...], "total_pages": int}
        """
        try:
            loader = get_loader(file_path)
            file_name, pages = await asyncio.to_thread(loader.load, file_path)

            return {
                "success": True,
                "file_name": file_name,
                "file_path": file_path,
                "pages": pages,
                "total_pages": len(pages),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
            }

    async def generate_from_page(
        self,
        technique: str,
        page_text: str,
        page_index: int,
        file_name: str,
        custom_template: Optional[str] = None,
        **generator_kwargs
    ) -> Dict[str, Any]:
        """
        Generate fine-tuning data from a single page.

        MCP Tool: generate_from_page
        Returns: {"success": bool, "data_points": [...], "count": int}
        """
        try:
            # Get template
            if custom_template:
                template = custom_template
            else:
                template = self.template_manager.get_template(technique)

            # Create pipeline
            pipeline = PipelineFactory.create(
                technique=technique,
                llm=self.llm_provider,
                prompt_template=template,
                parser=self.parser,
                debug=self.config.get("debug", False),
                **generator_kwargs
            )

            # Process single page
            page_data = {
                "markdown": page_text,
                "index": page_index
            }

            results = await pipeline.run(file_name, [page_data])

            # Convert to dicts
            data_dicts = [asdict(dp) for dp in results]

            return {
                "success": True,
                "technique": technique,
                "file_name": file_name,
                "page_index": page_index,
                "data_points": data_dicts,
                "count": len(data_dicts),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "technique": technique,
                "page_index": page_index,
            }

    async def generate_from_document(
        self,
        technique: str,
        file_path: str,
        custom_template: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        **generator_kwargs
    ) -> Dict[str, Any]:
        """
        Generate fine-tuning data from an entire document.

        MCP Tool: generate_from_document
        Returns: {"success": bool, "data_points": [...], "stats": {...}}
        """
        try:
            # Load document
            load_result = await self.load_document(file_path)
            if not load_result["success"]:
                return load_result

            pages = load_result["pages"]
            file_name = load_result["file_name"]

            # Filter pages if range specified
            if start_page is not None or end_page is not None:
                start = start_page or 0
                end = end_page or len(pages)
                pages = pages[start:end]

            # Get template
            if custom_template:
                template = custom_template
            else:
                template = self.template_manager.get_template(technique)

            # Create pipeline
            pipeline = PipelineFactory.create(
                technique=technique,
                llm=self.llm_provider,
                prompt_template=template,
                parser=self.parser,
                debug=self.config.get("debug", False),
                **generator_kwargs
            )

            # Process all pages
            results = await pipeline.run(file_name, pages)

            # Convert to dicts
            data_dicts = [asdict(dp) for dp in results]

            # Calculate stats
            stats = {
                "total_pages_processed": len(pages),
                "total_data_points": len(data_dicts),
                "avg_per_page": len(data_dicts) / len(pages) if pages else 0,
            }

            return {
                "success": True,
                "technique": technique,
                "file_name": file_name,
                "file_path": file_path,
                "data_points": data_dicts,
                "stats": stats,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "technique": technique,
                "file_path": file_path,
            }

    async def generate_batch(
        self,
        technique: str,
        file_paths: List[str],
        custom_template: Optional[str] = None,
        **generator_kwargs
    ) -> Dict[str, Any]:
        """
        Generate fine-tuning data from multiple documents.

        MCP Tool: generate_batch
        Returns: {"success": bool, "results": [...], "summary": {...}}
        """
        results = []
        all_data_points = []

        for file_path in file_paths:
            result = await self.generate_from_document(
                technique=technique,
                file_path=file_path,
                custom_template=custom_template,
                **generator_kwargs
            )
            results.append(result)

            if result["success"]:
                all_data_points.extend(result["data_points"])

        # Summary stats
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        return {
            "success": True,
            "technique": technique,
            "total_files": len(file_paths),
            "successful": successful,
            "failed": failed,
            "total_data_points": len(all_data_points),
            "results": results,
            "all_data_points": all_data_points,
        }

    async def export_dataset(
        self,
        data_points: List[Dict],
        output_path: str,
        format: str = "jsonl",
    ) -> Dict[str, Any]:
        """
        Export dataset to file.

        MCP Tool: export_dataset
        Returns: {"success": bool, "output_path": str, "format": str}
        """
        try:
            # Ensure directory exists
            output_file = Path(output_path)
            await asyncio.to_thread(
                output_file.parent.mkdir, parents=True, exist_ok=True
            )

            # Export based on format
            exporters = {
                "json": DatasetExporter.to_json,
                "jsonl": DatasetExporter.to_jsonl,
                "excel": DatasetExporter.to_excel,
                "csv": DatasetExporter.to_csv,
                "huggingface": DatasetExporter.to_huggingface,
            }

            if format not in exporters:
                return {
                    "success": False,
                    "error": f"Unknown format: {format}",
                    "available_formats": list(exporters.keys()),
                }

            await asyncio.to_thread(exporters[format], data_points, output_path)

            return {
                "success": True,
                "output_path": output_path,
                "format": format,
                "data_point_count": len(data_points),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output_path": output_path,
            }

    def list_techniques(self) -> Dict[str, Any]:
        """
        List available fine-tuning techniques.

        MCP Tool: list_techniques
        Returns: {"techniques": [...]}
        """
        from ..generators.registry import GENERATOR_REGISTRY

        return {
            "success": True,
            "techniques": list(GENERATOR_REGISTRY.keys()),
        }

    def get_technique_schema(self, technique: str) -> Dict[str, Any]:
        """
        Get the data schema for a technique.

        MCP Tool: get_technique_schema
        Returns: {"technique": str, "schema": {...}}
        """
        from ..generators.registry import GENERATOR_REGISTRY

        if technique not in GENERATOR_REGISTRY:
            return {
                "success": False,
                "error": f"Unknown technique: {technique}",
            }

        _, datapoint_class = GENERATOR_REGISTRY[technique]

        # Get fields from dataclass
        import dataclasses
        fields = dataclasses.fields(datapoint_class)

        schema = {
            field.name: {
                "type": str(field.type),
                "required": field.default == dataclasses.MISSING,
            }
            for field in fields
        }

        return {
            "success": True,
            "technique": technique,
            "schema": schema,
        }

    def get_template(self, technique: str) -> Dict[str, Any]:
        """
        Get the default prompt template for a technique.

        MCP Tool: get_template
        Returns: {"technique": str, "template": str}
        """
        try:
            template = self.template_manager.get_template(technique)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "technique": technique,
            }

        return {
            "success": True,
            "technique": technique,
            "template": template,
        }
