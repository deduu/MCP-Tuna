# providers/embeddings/openai.py
from ..base import EmbeddingProvider
from openai import OpenAI
import numpy as np


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        batch_size: int = 64,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts):
        vectors = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]

            resp = self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            vectors.extend([d.embedding for d in resp.data])

        return np.array(vectors)
