from agentsoul.providers.base import BaseLLM
from agentsoul.providers.openai import OpenAIProvider

try:
    from agentsoul.providers.hf import HuggingFaceProvider
except ImportError:
    HuggingFaceProvider = None  # torch/transformers not installed

try:
    from agentsoul.providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicProvider = None  # anthropic not installed

try:
    from agentsoul.providers.google import GoogleGeminiProvider
except ImportError:
    GoogleGeminiProvider = None  # google-genai not installed
