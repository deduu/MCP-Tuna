"""Composite data preparation MCP server.

Combines: extract, generate, clean, normalize, dataset tools.
No GPU required — only needs an LLM API key for generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentsoul.server import MCPServer
from shared.config import CleaningConfig, GeneratorConfig, NormalizationConfig


class DataPrepServer:
    """All data preparation tools in a single lightweight MCP server."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._generator_svc = None
        self._cleaning_svc = None
        self._normalization_svc = None
        self._dataset_svc = None
        self.mcp = MCPServer("mcp-tuna-data", "1.0.0")
        self._register_tools()

    # -- lazy accessors --
    @property
    def generator(self):
        if self._generator_svc is None:
            from data_generator_pipeline.services.pipeline_service import PipelineService
            from shared.provider_factory import create_llm
            gen_config = GeneratorConfig(**{
                k: v for k, v in self._config.get("generator", {}).items()
                if k in GeneratorConfig.model_fields
            })
            self._generator_svc = PipelineService(create_llm(gen_config), gen_config)
        return self._generator_svc

    @property
    def cleaner(self):
        if self._cleaning_svc is None:
            from data_cleaning_pipeline.services.cleaning_service import DataCleaningService
            self._cleaning_svc = DataCleaningService()
        return self._cleaning_svc

    @property
    def normalizer(self):
        if self._normalization_svc is None:
            from data_normalization_pipeline.services.normalization_service import DataNormalizationService
            self._normalization_svc = DataNormalizationService()
        return self._normalization_svc

    @property
    def dataset_service(self):
        if self._dataset_svc is None:
            from shared.dataset_service import DatasetService
            self._dataset_svc = DatasetService()
        return self._dataset_svc

    def _register_tools(self):
        self._register_extract_tools()
        self._register_generate_tools()
        self._register_clean_tools()
        self._register_normalize_tools()
        self._register_dataset_tools()

    # -- extract --
    def _register_extract_tools(self):
        @self.mcp.tool(name="extract.load_document",
                       description="Load and parse a document file (PDF, Markdown, DOCX)")
        async def load_document(file_path: str) -> str:
            from data_generator_pipeline.loaders import get_loader
            try:
                loader = get_loader(file_path)
                file_name, pages = loader.load(file_path)
                return json.dumps({"success": True, "file_name": file_name, "pages": pages, "total_pages": len(pages)}, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, indent=2)

    # -- generate --
    def _register_generate_tools(self):
        @self.mcp.tool(name="generate.from_document",
                       description="Generate fine-tuning data from an entire document")
        async def gen_from_doc(
            technique: str, file_path: str,
            custom_template: Optional[str] = None,
            start_page: Optional[int] = None, end_page: Optional[int] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_document(
                technique=technique, file_path=file_path,
                custom_template=custom_template, start_page=start_page, end_page=end_page,
            ), indent=2)

        @self.mcp.tool(name="generate.from_page",
                       description="Generate fine-tuning data from a single page")
        async def gen_from_page(
            technique: str, page_text: str, page_index: int, file_name: str,
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_page(
                technique=technique, page_text=page_text,
                page_index=page_index, file_name=file_name,
                custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(name="generate.batch",
                       description="Generate fine-tuning data from multiple documents")
        async def gen_batch(
            technique: str, file_paths: List[str],
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_batch(
                technique=technique, file_paths=file_paths, custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(name="generate.list_techniques",
                       description="List available fine-tuning techniques")
        async def list_techniques() -> str:
            from data_generator_pipeline.generators.registry import GENERATOR_REGISTRY
            return json.dumps({"success": True, "techniques": list(GENERATOR_REGISTRY.keys())}, indent=2)

        @self.mcp.tool(name="generate.get_schema",
                       description="Get the data schema for a fine-tuning technique")
        async def get_schema(technique: str) -> str:
            from data_generator_pipeline.generators.registry import GENERATOR_REGISTRY
            import dataclasses
            if technique not in GENERATOR_REGISTRY:
                return json.dumps({"success": False, "error": f"Unknown technique: {technique}"}, indent=2)
            _, dp_cls = GENERATOR_REGISTRY[technique]
            schema = {f.name: {"type": str(f.type), "required": f.default == dataclasses.MISSING} for f in dataclasses.fields(dp_cls)}
            return json.dumps({"success": True, "technique": technique, "schema": schema}, indent=2)

        @self.mcp.tool(name="generate.get_template",
                       description="Get the default prompt template for a fine-tuning technique")
        async def get_template(technique: str) -> str:
            return json.dumps(self.generator.get_template(technique), indent=2)

        @self.mcp.tool(name="generate.from_text",
                       description="Generate fine-tuning data from raw text (no file required)")
        async def gen_from_text(
            technique: str, text: str, source_name: str = "raw_text",
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_page(
                technique=technique, page_text=text,
                page_index=0, file_name=source_name, custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(name="generate.from_hf_dataset",
                       description="Load a dataset from HuggingFace Hub and return as data_points")
        async def gen_from_hf(
            dataset_name: str, split: str = "train",
            subset: Optional[str] = None, max_rows: Optional[int] = None,
            column_mapping: Optional[str] = None,
        ) -> str:
            try:
                from datasets import load_dataset as hf_load
                ds = hf_load(dataset_name, subset, split=split)
                if max_rows:
                    ds = ds.select(range(min(max_rows, len(ds))))
                mapping = json.loads(column_mapping) if column_mapping else {}
                data_points = []
                for row in ds:
                    dp = {}
                    for col, val in row.items():
                        target = mapping.get(col, col)
                        dp[target] = val if isinstance(val, (str, int, float, bool, list)) else str(val)
                    data_points.append(dp)
                return json.dumps({"success": True, "data_points": data_points, "count": len(data_points), "original_columns": list(ds.column_names)}, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, indent=2)

    # -- clean --
    def _register_clean_tools(self):
        @self.mcp.tool(name="clean.dataset", description="Run all cleaning steps on a dataset")
        async def clean_dataset(
            data_points: List[Dict], remove_duplicates: bool = True,
            min_instruction_length: int = 10, min_output_length: int = 20,
        ) -> str:
            config = CleaningConfig(remove_duplicates=remove_duplicates, min_instruction_length=min_instruction_length, min_output_length=min_output_length)
            return json.dumps(await self.cleaner.clean_dataset(data_points, config), indent=2)

        @self.mcp.tool(name="clean.deduplicate", description="Remove duplicate entries by key")
        async def deduplicate(data_points: List[Dict], key: str = "instruction") -> str:
            return json.dumps(await self.cleaner.deduplicate(data_points, key), indent=2)

        @self.mcp.tool(name="clean.validate_schema", description="Validate entries have required fields")
        async def validate_schema(data_points: List[Dict], technique: str = "sft") -> str:
            return json.dumps(await self.cleaner.validate_schema(data_points, technique), indent=2)

        @self.mcp.tool(name="clean.remove_short", description="Filter entries below length thresholds")
        async def remove_short(data_points: List[Dict], min_instruction: int = 10, min_output: int = 20) -> str:
            return json.dumps(await self.cleaner.remove_short_entries(data_points, min_instruction, min_output), indent=2)

    # -- normalize --
    def _register_normalize_tools(self):
        @self.mcp.tool(name="normalize.dataset", description="Apply all normalization steps")
        async def normalize_dataset(
            data_points: List[Dict], target_format: str = "sft",
            merge_instruction_input: bool = True, strip_whitespace: bool = True,
        ) -> str:
            config = NormalizationConfig(target_format=target_format, merge_instruction_input=merge_instruction_input, strip_whitespace=strip_whitespace)
            return json.dumps(await self.normalizer.normalize_dataset(data_points, config), indent=2)

        @self.mcp.tool(name="normalize.merge_fields", description="Combine instruction + input into single field")
        async def merge_fields(data_points: List[Dict]) -> str:
            return json.dumps(await self.normalizer.merge_instruction_input(data_points), indent=2)

        @self.mcp.tool(name="normalize.standardize_keys", description="Rename keys to target format")
        async def standardize_keys(data_points: List[Dict], target_format: str = "sft") -> str:
            return json.dumps(await self.normalizer.standardize_keys(data_points, target_format), indent=2)

        @self.mcp.tool(name="normalize.strip_text", description="Strip whitespace and normalize unicode")
        async def strip_text(data_points: List[Dict]) -> str:
            return json.dumps(await self.normalizer.strip_and_clean_text(data_points), indent=2)

    # -- dataset --
    def _register_dataset_tools(self):
        @self.mcp.tool(name="dataset.save", description="Save data_points to file (JSONL/JSON/Parquet)")
        async def dataset_save(data_points: List[Dict], output_path: str, format: str = "jsonl") -> str:
            return json.dumps(await self.dataset_service.save(data_points, output_path, format), indent=2)

        @self.mcp.tool(name="dataset.load", description="Load dataset from file, return data_points")
        async def dataset_load(file_path: str) -> str:
            return json.dumps(await self.dataset_service.load(file_path), indent=2)

        @self.mcp.tool(name="dataset.preview", description="Show first N rows from a dataset file")
        async def dataset_preview(file_path: str, n: int = 5) -> str:
            return json.dumps(await self.dataset_service.preview(file_path, n), indent=2)

        @self.mcp.tool(name="dataset.info", description="Get metadata about a dataset file")
        async def dataset_info(file_path: str) -> str:
            return json.dumps(await self.dataset_service.info(file_path), indent=2)

        @self.mcp.tool(name="dataset.delete", description="Delete a dataset file")
        async def dataset_delete(file_path: str) -> str:
            return json.dumps(await self.dataset_service.delete(file_path), indent=2)

        @self.mcp.tool(name="dataset.split", description="Split into train/val/test files")
        async def dataset_split(
            file_path: str, output_dir: str,
            train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
            seed: int = 42,
        ) -> str:
            return json.dumps(await self.dataset_service.split(file_path, output_dir, train_ratio, val_ratio, test_ratio, seed), indent=2)

        @self.mcp.tool(name="dataset.merge", description="Merge multiple dataset files into one")
        async def dataset_merge(
            file_paths: List[str], output_path: str,
            deduplicate: bool = False, dedup_key: str = "instruction",
        ) -> str:
            return json.dumps(await self.dataset_service.merge(file_paths, output_path, deduplicate, dedup_key), indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
