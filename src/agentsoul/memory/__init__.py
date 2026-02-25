from .base import BaseMemory
from .conversation import ConversationMemory
from .composite import CompositeMemory

__all__ = [
    "BaseMemory",
    "ConversationMemory",
    "CompositeMemory",
]

# ChromaMemory requires chromadb — import lazily
try:
    from .chroma import ChromaMemory
    __all__.append("ChromaMemory")
except ImportError:
    pass

# PostgresMemory adapter — no hard deps (store owns the real deps)
from .postgres import PostgresMemory
__all__.append("PostgresMemory")
