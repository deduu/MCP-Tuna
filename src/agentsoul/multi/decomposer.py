"""
Query Decomposer — parallel sub-query fan-out for complex questions.

Decomposes a complex user query into independent sub-queries, fans them
out to a worker agent in parallel via ``asyncio.gather``, and synthesizes
a unified answer.

Simple queries are detected by a fast heuristic and passed directly to
the worker without decomposition overhead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from agentsoul.core.agent import AgentSoul
    from agentsoul.providers.base import BaseLLM

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass
class DecompositionResult:
    """Result of a query decomposition execution."""

    original_query: str
    was_decomposed: bool = False
    sub_queries: List[str] = field(default_factory=list)
    sub_results: List[str] = field(default_factory=list)
    final_answer: str = ""
    timings: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------
# Complexity heuristic
# ------------------------------------------------------------------


def _is_complex_query(query: str, min_words: int = 8) -> bool:
    """Heuristic check whether a query likely benefits from decomposition.

    A query is considered complex if it meets ANY of:
    - Contains multiple question marks
    - Contains comparison keywords (compare, versus, vs, difference between)
    - Contains enumeration keywords (what are the main, list the, ...)
    - Contains multi-topic conjunctions in longer queries
    - Has more than *min_words* words (baseline length gate)

    Short greetings and single-fact questions return ``False``.
    """
    words = query.split()
    if len(words) < min_words:
        return False

    lower = query.lower()

    if query.count("?") > 1:
        return True

    comparison_markers = [
        "compare", "versus", "vs.", "vs ",
        "difference between", "differences between",
        "pros and cons", "advantages and disadvantages",
    ]
    if any(m in lower for m in comparison_markers):
        return True

    breadth_markers = [
        "what are the main", "what are the key",
        "list the", "summarize the",
        "overview of", "explain the different",
    ]
    if any(m in lower for m in breadth_markers):
        return True

    if len(words) >= 12:
        conjunction_markers = [
            " and also ", " as well as ", " additionally ",
            " furthermore ", " in addition to ",
        ]
        if any(m in lower for m in conjunction_markers):
            return True

    return False


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_DECOMPOSE_SYSTEM_PROMPT = """\
You are a query decomposition specialist. Your job is to break a complex user \
question into independent, non-overlapping sub-queries that can be researched \
in parallel.

Rules:
1. Each sub-query must be self-contained and understandable without context from the others.
2. Sub-queries should cover ALL aspects of the original question.
3. Do NOT create redundant or overlapping sub-queries.
4. If the question is already simple and focused, return it as a single sub-query.
5. Aim for 2-5 sub-queries. Never exceed 7.
6. Each sub-query should be a clear, specific question or research task.

Respond with ONLY valid JSON in this exact format:
{"sub_queries": ["sub-query 1", "sub-query 2", ...]}

Do not include any text outside the JSON object."""

_DECOMPOSE_USER_TEMPLATE = (
    "Decompose this question into independent sub-queries:\n\n{query}"
)

_SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert synthesizer. You will receive the user's original question \
and research results from multiple independent sub-queries.

Your task:
1. Combine ALL the sub-results into a single, coherent, comprehensive answer.
2. Cross-reference information across sub-results to find patterns and connections.
3. Resolve any contradictions by noting them explicitly.
4. Use proper Markdown formatting (headings, bold, lists) for readability.
5. Cite sources naturally where available.
6. Do NOT mention the decomposition process or sub-queries.
7. Answer the original question directly and substantively."""

_SYNTHESIS_USER_TEMPLATE = """\
# Original Question
{query}

# Research Results

{sub_results}

Now synthesize these results into a comprehensive answer to the original question."""


# ------------------------------------------------------------------
# QueryDecomposer
# ------------------------------------------------------------------


class QueryDecomposer:
    """Decomposes complex queries into sub-queries, fans out to a worker
    agent in parallel, and synthesizes a unified answer.

    Usage::

        decomposer = QueryDecomposer(
            llm=my_llm,
            worker=my_research_agent,
        )
        result = await decomposer.run(
            "Compare the economic policies of Japan and Germany "
            "and explain how they affect tech innovation"
        )
        print(result.final_answer)

    For simple queries the decomposition step is automatically skipped,
    so there is zero overhead compared to calling the worker directly.
    """

    def __init__(
        self,
        llm: "BaseLLM",
        worker: "AgentSoul",
        *,
        decompose_system_prompt: Optional[str] = None,
        synthesis_system_prompt: Optional[str] = None,
        min_complexity_words: int = 8,
        max_sub_queries: int = 5,
        skip_decomposition: bool = False,
        force_decomposition: bool = False,
    ):
        """
        Args:
            llm: LLM provider used for the decomposition and synthesis
                 calls.  Can be the same provider as the worker or a
                 cheaper / faster one.
            worker: :class:`AgentSoul` instance that handles each
                    sub-query.  The same instance is reused (each
                    ``invoke()`` call is independent).
            decompose_system_prompt: Override the default decomposition prompt.
            synthesis_system_prompt: Override the default synthesis prompt.
            min_complexity_words: Minimum word count for the complexity
                                  heuristic (default 8).
            max_sub_queries: Cap on the number of sub-queries (default 5).
            skip_decomposition: Always skip decomposition (pass-through).
            force_decomposition: Always decompose, ignoring the heuristic.
        """
        self.llm = llm
        self.worker = worker
        self.decompose_system_prompt = decompose_system_prompt or _DECOMPOSE_SYSTEM_PROMPT
        self.synthesis_system_prompt = synthesis_system_prompt or _SYNTHESIS_SYSTEM_PROMPT
        self.min_complexity_words = min_complexity_words
        self.max_sub_queries = max_sub_queries
        self.skip_decomposition = skip_decomposition
        self.force_decomposition = force_decomposition

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: str) -> DecompositionResult:
        """Execute the full decomposition pipeline.

        Returns a :class:`DecompositionResult` with the final answer,
        sub-query details, and timing information.
        """
        total_start = time.perf_counter()
        result = DecompositionResult(original_query=query)

        # --- Step 0: complexity check ---
        should_decompose = self.force_decomposition or (
            not self.skip_decomposition
            and _is_complex_query(query, min_words=self.min_complexity_words)
        )

        if not should_decompose:
            logger.info("[DECOMPOSER] Simple query, skipping decomposition")
            result.was_decomposed = False
            result.final_answer = await self.worker.invoke(query)
            result.timings["total_time"] = time.perf_counter() - total_start
            return result

        # --- Step 1: decompose ---
        logger.info("[DECOMPOSER] Complex query, decomposing...")
        decompose_start = time.perf_counter()
        sub_queries = await self._decompose(query)
        result.timings["decompose_time"] = time.perf_counter() - decompose_start
        result.sub_queries = sub_queries
        result.was_decomposed = True
        logger.info("[DECOMPOSER] Decomposed into %d sub-queries", len(sub_queries))

        # Edge case: single sub-query → skip synthesis overhead
        if len(sub_queries) == 1:
            logger.info("[DECOMPOSER] Single sub-query, running without synthesis")
            answer = await self.worker.invoke(sub_queries[0])
            result.sub_results = [answer]
            result.final_answer = answer
            result.timings["total_time"] = time.perf_counter() - total_start
            return result

        # --- Step 2: parallel fan-out ---
        logger.info("[DECOMPOSER] Fanning out %d sub-queries in parallel", len(sub_queries))
        fan_out_start = time.perf_counter()
        sub_results = await self._fan_out(sub_queries)
        result.timings["fan_out_time"] = time.perf_counter() - fan_out_start
        result.sub_results = sub_results

        # --- Step 3: synthesize ---
        logger.info("[DECOMPOSER] Synthesizing results")
        synthesis_start = time.perf_counter()
        result.final_answer = await self._synthesize(query, sub_queries, sub_results)
        result.timings["synthesis_time"] = time.perf_counter() - synthesis_start

        result.timings["total_time"] = time.perf_counter() - total_start
        logger.info(
            "[DECOMPOSER] Complete. Timings: %s",
            json.dumps({k: round(v, 3) for k, v in result.timings.items()}),
        )
        return result

    async def invoke(self, query: str) -> str:
        """Convenience method that returns only the final answer string."""
        result = await self.run(query)
        return result.final_answer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _decompose(self, query: str) -> List[str]:
        """Single LLM call to decompose the query into sub-queries."""
        messages = [
            {"role": "system", "content": self.decompose_system_prompt},
            {"role": "user", "content": _DECOMPOSE_USER_TEMPLATE.format(query=query)},
        ]

        response = await self.llm.chat(messages, tools=None, enable_thinking=False)
        content = (response.content or "").strip()

        try:
            # Handle markdown code-block wrapper
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            sub_queries = parsed.get("sub_queries", [])
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(
                "[DECOMPOSER] Failed to parse decomposition JSON: %s. "
                "Falling back to original query.",
                e,
            )
            return [query]

        if not sub_queries or not isinstance(sub_queries, list):
            return [query]

        # Enforce cap and ensure strings
        return [str(q) for q in sub_queries[: self.max_sub_queries]]

    async def _fan_out(self, sub_queries: List[str]) -> List[str]:
        """Execute all sub-queries concurrently via ``asyncio.gather``.

        Failed sub-queries return an error string rather than crashing
        the pipeline, following the pattern in
        :class:`~agentsoul.retrieval.research_service.ResearchToolService`.
        """

        async def _safe_invoke(sub_query: str) -> str:
            try:
                return await self.worker.invoke(sub_query)
            except Exception as e:
                logger.warning(
                    "[DECOMPOSER] Sub-query failed: %s — %s",
                    sub_query[:80], e,
                )
                return f"[Error researching: {sub_query[:80]}]: {e}"

        tasks = [_safe_invoke(sq) for sq in sub_queries]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _synthesize(
        self,
        original_query: str,
        sub_queries: List[str],
        sub_results: List[str],
    ) -> str:
        """Single LLM call to merge sub-results into a coherent answer."""
        parts = []
        for i, (sq, sr) in enumerate(zip(sub_queries, sub_results), 1):
            parts.append(f"## Research {i}: {sq}\n\n{sr}")

        sub_results_text = "\n\n---\n\n".join(parts)

        messages = [
            {"role": "system", "content": self.synthesis_system_prompt},
            {"role": "user", "content": _SYNTHESIS_USER_TEMPLATE.format(
                query=original_query,
                sub_results=sub_results_text,
            )},
        ]

        response = await self.llm.chat(messages, tools=None, enable_thinking=False)
        return (response.content or "Could not synthesize an answer.").strip()
