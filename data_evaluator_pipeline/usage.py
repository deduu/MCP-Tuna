"""Example usage of the evaluator pipeline with shared config."""

from shared.config import EvaluatorConfig
from shared.providers import SyncLLMAdapter
from shared.provider_factory import create_llm

from .core.metrics.complexity import ComplexityMetric
from .core.metrics.ifd import InstructionFollowingDifficultyMetric
from .core.metrics.quality import LLMQualityMetric
from .core.evaluator import MetricEvaluator

from .selection.quality import QualityThresholdSelector
from .io.loaders import load_jsonl
from .analysis.dataset_analyzer import DatasetAnalyzer
from .pipeline import InstructionTuningPipeline


def run(data_path: str = "data/sft/data_sft.jsonl", config: EvaluatorConfig = None):
    """Run the evaluator pipeline with the given config."""
    if config is None:
        config = EvaluatorConfig()

    # Create LLM via shared factory → wrap for sync usage
    llm = SyncLLMAdapter(create_llm(config))

    # Metrics
    metrics = [
        ComplexityMetric(language_code=config.language),
        InstructionFollowingDifficultyMetric(),
        LLMQualityMetric(llm),
    ]

    evaluator = MetricEvaluator(metrics, config.weights)
    selector = QualityThresholdSelector(evaluator, threshold=config.threshold)
    analyzer = DatasetAnalyzer(evaluator)

    pipeline = InstructionTuningPipeline(evaluator, selector, analyzer)

    dataset = load_jsonl(data_path)
    result = pipeline.run(dataset)

    print(result["original"], "→", result["selected"])
    return result


if __name__ == "__main__":
    run()
