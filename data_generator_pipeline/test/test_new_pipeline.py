# ============================================================================
# FILE: scripts/run_pipeline.py
# ============================================================================

import asyncio
import os
import argparse
from pathlib import Path
import yaml
from dotenv import load_dotenv

from ..core.factory import PipelineFactory
from ..parsers.json_extractor import JsonExtractor
from ..prompts.templates import PromptTemplateManager
from ..exporters.dataset import DatasetExporter
from ..loaders import get_loader
from src.agent_framework.providers.openai import OpenAIProvider
from src.agent_framework.config import settings


async def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning data generation pipeline")
    parser.add_argument(
        "--technique",
        type=str,
        required=True,
        choices=["sft", "dpo", "grpo"],
        help="Fine-tuning technique"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./AgentY/data_generator_pipeline/scripts/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="jsonl",
        choices=["json", "jsonl", "excel", "csv", "huggingface"],
        help="Output format"
    )
    parser.add_argument(
        "--max-items-per-page",
        type=int,
        default=None,
        help="Maximum number of data points to generate per page (None for unlimited)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    load_dotenv(override=True)  # override system env if needed
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    print("\n[DEBUG] OpenAIProvider fixture values")
    print("  OPENAI_MODEL =", model)
    print("  OPENAI_API_BASE =", base_url)
    print("  OPENAI_API_KEY =", api_key[:6] + "..." if api_key else None)
    # Setup components
    # llm = OpenAIProvider(
    #     model=config["model"],
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )
    llm = OpenAIProvider(model=model, api_key=api_key, base_url=base_url)

    parser_obj = JsonExtractor()
    template_manager = PromptTemplateManager()
    prompt_template = template_manager.get_template(args.technique)

    # Create pipeline
    pipeline = PipelineFactory.create(
        technique=args.technique,
        llm=llm,
        prompt_template=prompt_template,
        parser=parser_obj,
        debug=args.debug,
        **config.get("generator_kwargs", {})
    )

    # Process files
    file_paths = settings.MD_DATA_PATH
    all_data_points = []

    for file_path in file_paths:
        print(f"\n📄 Processing {file_path}...")
        loader = get_loader(file_path)
        file_name, pages = loader.load(file_path)

        data_points = await pipeline.run(file_name, pages, max_items_per_page=args.max_items_per_page)
        all_data_points.extend(data_points)

        print(f"✅ Generated {len(data_points)} data points from {file_name}")

    # Export results
    output_dir = Path(config["output_dir"]) / args.technique
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dicts = pipeline.to_dict_list(all_data_points)

    output_file = output_dir / f"data_{args.technique}.{args.output_format}"

    exporters = {
        "json": DatasetExporter.to_json,
        "jsonl": DatasetExporter.to_jsonl,
        "excel": DatasetExporter.to_excel,
        "csv": DatasetExporter.to_csv,
        "huggingface": DatasetExporter.to_huggingface,
    }

    exporters[args.output_format](data_dicts, str(output_file))

    print(f"\n✨ Done! Generated {len(all_data_points)} total data points")
    print(f"📁 Saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
