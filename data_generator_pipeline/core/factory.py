# ============================================================================
# FILE: src/finetuning/core/factory.py
# ============================================================================

from typing import Dict, Type, Tuple
from .base import BaseGenerator, BaseParser, BaseLLM
from .pipeline import FineTuningPipeline
from ..models.datapoints import BaseDataPoint
from ..generators.registry import GENERATOR_REGISTRY, register_generator


class PipelineFactory:
    """Factory to create pipelines for different fine-tuning techniques."""

    @classmethod
    def create(
        cls,
        technique: str,
        llm: "BaseLLM",
        prompt_template: str,
        parser: BaseParser,
        debug: bool = False,
        **generator_kwargs,
    ) -> FineTuningPipeline:
        """Create a pipeline for the specified technique."""
        if technique not in GENERATOR_REGISTRY:
            available = list(GENERATOR_REGISTRY.keys())
            raise ValueError(
                f"Unknown technique: '{technique}'. "
                f"Available: {available}"
            )

        generator_class, data_point_class = GENERATOR_REGISTRY[technique]

        # Filter generator_kwargs based on technique
        filtered_kwargs = {}
        if technique == "grpo":
            filtered_kwargs = {
                k: v for k, v in generator_kwargs.items() if k == 'num_responses'}

        generator = generator_class(
            llm=llm,
            prompt_template=prompt_template,
            parser=parser,
            debug=debug,
            **filtered_kwargs,
        )

        return FineTuningPipeline(generator, data_point_class)
