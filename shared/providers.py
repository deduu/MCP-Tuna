"""Bridge to the agent framework's BaseLLM + sync adapter for evaluator metrics."""

import asyncio
from typing import List, Dict, Any

from agentsoul.providers.base import BaseLLM as AgentBaseLLM

# Re-export so all Transcendence code imports BaseLLM from one place
BaseLLM = AgentBaseLLM


class SyncLLMAdapter:
    """Wraps an async BaseLLM for evaluator metrics that require synchronous calls."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, prompt: str) -> str:
        """Synchronous generate — runs the async chat() in an event loop."""
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context — create a new thread to avoid deadlock
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                resp = pool.submit(asyncio.run, self.llm.chat(messages)).result()
        else:
            resp = asyncio.run(self.llm.chat(messages))

        return resp.content or ""
