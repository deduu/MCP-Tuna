"""Perplexity metric via OpenAI Chat Completions logprobs.

Uses direct AsyncOpenAI client (not agentsoul's BaseLLM) because
BaseLLM does not expose logprobs. Falls back to env vars if
api_key / api_base are not provided.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


async def compute_perplexity(
    question: str,
    generated: str,
    reference: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute perplexity of the generated answer given the question context.

    Sends a chat completion request with ``logprobs=True`` and computes:
      - ``mean_logprob``: average token log-probability
      - ``perplexity``: exp(-mean_logprob)
      - ``num_tokens``: number of tokens scored

    Returns zeros gracefully when the API does not support logprobs or on error.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    api_base = api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base

    try:
        client = AsyncOpenAI(**client_kwargs)

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are evaluating a model's answer. "
                        "Reproduce the following answer exactly:\n\n"
                        f"Reference: {reference}"
                    ),
                },
                {"role": "user", "content": question},
            ],
            logprobs=True,
            max_tokens=1024,
            temperature=0.0,
        )

        choice = response.choices[0]
        if choice.logprobs is None or choice.logprobs.content is None:
            return {"perplexity": 0.0, "mean_logprob": 0.0, "num_tokens": 0}

        logprob_values = [token.logprob for token in choice.logprobs.content]

        if not logprob_values:
            return {"perplexity": 0.0, "mean_logprob": 0.0, "num_tokens": 0}

        mean_logprob = sum(logprob_values) / len(logprob_values)
        perplexity = math.exp(-mean_logprob)

        return {
            "perplexity": perplexity,
            "mean_logprob": mean_logprob,
            "num_tokens": len(logprob_values),
        }

    except Exception:
        return {"perplexity": 0.0, "mean_logprob": 0.0, "num_tokens": 0}
