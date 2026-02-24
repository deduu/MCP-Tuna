"""Single factory for creating LLM providers from config."""

import os
from .config import PipelineConfig
from .providers import BaseLLM


def create_llm(config: PipelineConfig) -> BaseLLM:
    """Create the appropriate LLM provider based on model name in config."""
    model = config.model

    if model.startswith("gpt") or model.startswith("o"):
        from agent_framework.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model_id=model,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    else:
        from agent_framework.providers.hf import HuggingFaceProvider
        return HuggingFaceProvider(model_path=model)
