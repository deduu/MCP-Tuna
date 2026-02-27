"""Judge registry and auto-registration for LLM-as-a-judge types."""
from __future__ import annotations

from shared.registry import judge_registry

# Import built-in judges to trigger auto-registration via BaseJudge.__init_subclass__
from .pointwise import PointwiseJudge  # noqa: F401
from .pairwise import PairwiseJudge  # noqa: F401
from .reference_free import ReferenceFreeJudge  # noqa: F401
from .rubric import RubricJudge  # noqa: F401

__all__ = ["judge_registry"]
