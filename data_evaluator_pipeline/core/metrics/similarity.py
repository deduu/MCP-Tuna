from .base import BaseMetric
from providers.base import EmbeddingProvider
import numpy as np


class SemanticSimilarityMetric(BaseMetric):
    name = "semantic_similarity"

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder

    def compute(self, dp):
        vecs = self.embedder.embed(
            [dp.full_instruction, dp.output]
        )
        return float(np.dot(vecs[0], vecs[1]) / (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        ))
