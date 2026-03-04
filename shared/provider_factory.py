"""Single factory for creating LLM providers from config."""

import os
from .config import PipelineConfig
from .providers import BaseLLM


def create_llm(config: PipelineConfig) -> BaseLLM:
    """Create the appropriate LLM provider based on model name in config."""
    model = config.model

    if model.startswith("gpt") or model.startswith("o"):
        from agentsoul.providers.openai import OpenAIProvider
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_API_BASE_URL")
            or None
        )
        return OpenAIProvider(
            model_id=model,
            base_url=base_url,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    elif model.startswith("claude"):
        from agentsoul.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            model_id=model,
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            base_url=os.getenv("ANTHROPIC_API_BASE"),
        )
    elif model.startswith("gemini"):
        from agentsoul.providers.google import GoogleGeminiProvider
        return GoogleGeminiProvider(
            model_id=model,
            api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
    else:
        from agentsoul.providers.hf import HuggingFaceProvider
        return HuggingFaceProvider(model_path=model)
