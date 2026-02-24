# ============================================================================
# FILE: src/finetuning/core/factory.py
# ============================================================================

from .base import BaseParser, BaseLLM
from .pipeline import FineTuningPipeline
from ..generators.registry import GENERATOR_REGISTRY


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

        # Let each generator class decide which kwargs it accepts
        filtered_kwargs = generator_class.filter_kwargs(generator_kwargs)

        generator = generator_class(
            llm=llm,
            prompt_template=prompt_template,
            parser=parser,
            debug=debug,
            **filtered_kwargs,
        )

        return FineTuningPipeline(generator, data_point_class)
