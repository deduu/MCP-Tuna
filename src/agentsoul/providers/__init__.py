from agentsoul.providers.base import BaseLLM
from agentsoul.providers.openai import OpenAIProvider

try:
    from agentsoul.providers.hf import HuggingFaceProvider
except ImportError:
    HuggingFaceProvider = None  # torch/transformers not installed
