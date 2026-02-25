"""
agentsoul.retrieval — Hybrid retrieval module (FAISS + BM25).

Core utilities (tokenizer, fusion) are always available.
Service classes require optional dependencies:
    pip install agentsoul[retrieval]
"""

from .tokenizer import simple_tokenize
from .fusion import base_fusion, reciprocal_rank_fusion, build_llm_context

__all__ = [
    "simple_tokenize",
    "base_fusion",
    "reciprocal_rank_fusion",
    "build_llm_context",
]

# BM25Service requires rank_bm25
try:
    from .bm25_service import BM25Service, BM25Pack, BM25Runtime
    __all__.extend(["BM25Service", "BM25Pack", "BM25Runtime"])
except ImportError:
    pass

# FAISSService requires faiss-cpu + langchain
try:
    from .faiss_service import FAISSService
    __all__.extend(["FAISSService"])
except ImportError:
    pass

# RAGToolService requires both BM25 and FAISS
try:
    from .rag_tool_service import RAGToolService
    __all__.extend(["RAGToolService"])
except ImportError:
    pass

# ResearchToolService has no hard deps beyond ToolService (sources are optional)
from .research_service import ResearchToolService
__all__.append("ResearchToolService")
