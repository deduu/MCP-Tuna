# providers/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
