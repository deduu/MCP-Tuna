from openai import OpenAI
import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_output_tokens=8192,
        )
        return response.output_text
