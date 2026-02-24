# providers/base.py — unified provider interfaces
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from AgentY.shared.providers import SyncLLMAdapter  # noqa: F401


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass


# Backwards-compatible alias so existing `from ..providers.base import LLMProvider` still works.
# SyncLLMAdapter exposes the same .generate(prompt) -> str interface.
LLMProvider = SyncLLMAdapter
