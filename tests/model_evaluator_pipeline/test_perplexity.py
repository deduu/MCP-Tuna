"""Unit tests for the perplexity metric."""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_logprobs_response(logprob_values: list[float]):
    """Build a mock ChatCompletion with logprobs on the content tokens."""
    tokens = []
    for lp in logprob_values:
        token = MagicMock()
        token.logprob = lp
        tokens.append(token)

    choice = MagicMock()
    choice.logprobs.content = tokens

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_no_logprobs_response():
    """Build a mock ChatCompletion where logprobs are not available."""
    choice = MagicMock()
    choice.logprobs = None

    response = MagicMock()
    response.choices = [choice]
    return response


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

class TestComputePerplexity:
    @pytest.mark.asyncio
    async def test_computation_math(self):
        """Perplexity = exp(-mean(logprobs))."""
        from model_evaluator_pipeline.metrics.perplexity import compute_perplexity

        logprob_values = [-0.5, -1.0, -1.5]
        expected_mean = sum(logprob_values) / len(logprob_values)
        expected_ppl = math.exp(-expected_mean)

        mock_response = _make_logprobs_response(logprob_values)

        with patch("model_evaluator_pipeline.metrics.perplexity.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client

            result = await compute_perplexity(
                question="What is X?",
                generated="X is Y.",
                reference="X is Y.",
                model="gpt-4o",
                api_key="test-key",
            )

        assert result["perplexity"] == pytest.approx(expected_ppl, rel=1e-4)
        assert result["mean_logprob"] == pytest.approx(expected_mean, rel=1e-4)
        assert result["num_tokens"] == 3

    @pytest.mark.asyncio
    async def test_no_logprobs_fallback(self):
        """Returns zeros when API doesn't support logprobs."""
        from model_evaluator_pipeline.metrics.perplexity import compute_perplexity

        mock_response = _make_no_logprobs_response()

        with patch("model_evaluator_pipeline.metrics.perplexity.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client

            result = await compute_perplexity(
                question="Q", generated="A", reference="R",
                model="gpt-4o", api_key="test-key",
            )

        assert result["perplexity"] == 0.0
        assert result["mean_logprob"] == 0.0
        assert result["num_tokens"] == 0

    @pytest.mark.asyncio
    async def test_env_var_fallback(self):
        """Falls back to env vars when api_key/api_base not provided."""
        from model_evaluator_pipeline.metrics.perplexity import compute_perplexity

        mock_response = _make_logprobs_response([-1.0])

        with patch("model_evaluator_pipeline.metrics.perplexity.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client

            with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key", "OPENAI_BASE_URL": "http://localhost"}):
                result = await compute_perplexity(
                    question="Q", generated="A", reference="R",
                    model="gpt-4o",
                )

            # Should have been called — client was constructed
            MockClient.assert_called_once()

        assert result["num_tokens"] == 1

    @pytest.mark.asyncio
    async def test_return_keys(self):
        """Result always contains perplexity, mean_logprob, num_tokens."""
        from model_evaluator_pipeline.metrics.perplexity import compute_perplexity

        mock_response = _make_logprobs_response([-0.2, -0.3])

        with patch("model_evaluator_pipeline.metrics.perplexity.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client

            result = await compute_perplexity(
                question="Q", generated="A", reference="R",
                model="gpt-4o", api_key="k",
            )

        assert "perplexity" in result
        assert "mean_logprob" in result
        assert "num_tokens" in result
        assert isinstance(result["perplexity"], float)
        assert isinstance(result["mean_logprob"], float)
        assert isinstance(result["num_tokens"], int)
