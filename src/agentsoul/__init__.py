"""
agentsoul: A modular, provider-agnostic AI agent framework with MCP tool support.
"""

from shared.version import get_package_version

__version__ = get_package_version()

from agentsoul.core.agent import AgentSoul
from agentsoul.core.message import MessageFormatter
from agentsoul.core.models import (
    LLMResponse,
    Message,
    MessageRole,
    ReflectionMode,
    ReflectionPolicy,
    StreamChunk,
    ToolCall,
)
from agentsoul.core.strategies import (
    CodeActExecStrategy,
    CodeActStrategy,
    JsonSchemaStrategy,
)
from agentsoul.core.tool_strategy import ToolCallingStrategy
from agentsoul.multi import AgentTeam, AgentTool, DecompositionResult, QueryDecomposer
from agentsoul.providers.base import BaseLLM
from agentsoul.providers.openai import OpenAIProvider
from agentsoul.tools.base import BaseToolService
from agentsoul.tools.composite import CompositeToolService
from agentsoul.tools.orchestrator import ToolOrchestrator
from agentsoul.tools.service import ToolService

# Memory system
from agentsoul.memory import BaseMemory, CompositeMemory, ConversationMemory, PostgresMemory

__all__ = [
    "AgentSoul",
    "BaseLLM",
    "OpenAIProvider",
    "ToolCall",
    "Message",
    "MessageRole",
    "LLMResponse",
    "StreamChunk",
    "ReflectionMode",
    "ReflectionPolicy",
    "MessageFormatter",
    "BaseToolService",
    "ToolService",
    "ToolOrchestrator",
    "CompositeToolService",
    "ToolCallingStrategy",
    "JsonSchemaStrategy",
    "CodeActStrategy",
    "CodeActExecStrategy",
    "BaseMemory",
    "ConversationMemory",
    "CompositeMemory",
    "PostgresMemory",
    "AgentTool",
    "AgentTeam",
    "QueryDecomposer",
    "DecompositionResult",
]

# ChromaMemory requires chromadb - expose if available.
try:
    from agentsoul.memory import ChromaMemory

    __all__.append("ChromaMemory")
except ImportError:
    pass

# Retrieval module - expose if optional deps are available.
try:
    from agentsoul.retrieval import BM25Service, FAISSService, RAGToolService

    __all__.extend(["BM25Service", "FAISSService", "RAGToolService"])
except ImportError:
    pass

# ResearchToolService - no hard deps beyond optional retrieval sources.
from agentsoul.retrieval import ResearchToolService

__all__.append("ResearchToolService")
