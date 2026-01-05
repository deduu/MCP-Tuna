import asyncio
import os
from ..loaders import get_loader
from ..sft_generator import SFTGenerator
from ..llm_clients import OpenAIClient
from ..prompt_template_registry import PromptTemplateRegistry
from ..parsers_extractor import JsonExtractor
from ..data_exporter import DatasetExporter
from ..pipeline import SFTPipeline
from src.agent_framework.config import settings
from src.agent_framework.providers.openai import OpenAIProvider
from ..PROMPT_SAMPLE import prompt_template

file_path = settings.MD_DATA_PATH


async def main():
    file_path = settings.MD_DATA_PATH

    # Setup pipeline
    prompt_registry = PromptTemplateRegistry(prompt_template)
    llm = OpenAIProvider(           # ⚠️ must be BaseLLM, not OpenAIClient
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    parser = JsonExtractor()
    generator = SFTGenerator(llm, prompt_registry, parser, debug=True)
    pipeline = SFTPipeline(generator)

    data_sft = []

    for file in file_path:
        print(f"Processing {file}...")
        loader = get_loader(file)
        file_name, pages = loader.load(file)

        # ✅ THIS IS THE AWAIT
        data = await pipeline.run(file_name, pages)
        data_sft.extend(data)

    output_dir = os.path.join(os.getcwd(), "data", "sft")
    os.makedirs(output_dir, exist_ok=True)

    DatasetExporter.to_excel(
        data_sft,
        os.path.join(output_dir, "data_sft.xlsx"),
    )


if __name__ == "__main__":
    asyncio.run(main())
