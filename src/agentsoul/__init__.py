"""
agentsoul: A modular, provider-agnostic AI agent framework with MCP tool support.
"""

__version__ = "0.1.0"

from agentsoul.core.agent import AgentSoul
from agentsoul.core.models import ToolCall, Message, LLMResponse, StreamChunk, MessageRole, ReflectionMode, ReflectionPolicy
from agentsoul.core.message import MessageFormatter
from agentsoul.providers.base import BaseLLM
from agentsoul.providers.openai import OpenAIProvider
from agentsoul.tools.base import BaseToolService
from agentsoul.tools.service import ToolService
from agentsoul.tools.orchestrator import ToolOrchestrator
from agentsoul.tools.composite import CompositeToolService
from agentsoul.core.tool_strategy import ToolCallingStrategy
from agentsoul.core.strategies import (
    JsonSchemaStrategy,
    CodeActStrategy,
    CodeActExecStrategy,
)

# Memory system
from agentsoul.memory import BaseMemory, ConversationMemory, CompositeMemory, PostgresMemory

# Multi-agent coordination
from agentsoul.multi import AgentTool, AgentTeam, QueryDecomposer, DecompositionResult

__all__ = [
    # Core
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
    # Memory
    "BaseMemory",
    "ConversationMemory",
    "CompositeMemory",
    "PostgresMemory",
    # Multi-agent
    "AgentTool",
    "AgentTeam",
    "QueryDecomposer",
    "DecompositionResult",
]

# ChromaMemory requires chromadb — expose if available
try:
    from agentsoul.memory import ChromaMemory
    __all__.append("ChromaMemory")
except ImportError:
    pass

# Retrieval module — expose if optional deps are available
try:
    from agentsoul.retrieval import BM25Service, FAISSService, RAGToolService
    __all__.extend(["BM25Service", "FAISSService", "RAGToolService"])
except ImportError:
    pass

# ResearchToolService — no hard deps (all sources optional)
from agentsoul.retrieval import ResearchToolService
__all__.append("ResearchToolService")
