from .providers.embeddings.openai import OpenAIEmbeddingProvider
from .providers.llm.openai import OpenAILLMProvider

from .core.metrics.complexity import ComplexityMetric
from .core.metrics.ifd import InstructionFollowingDifficultyMetric
from .core.metrics.quality import LLMQualityMetric
from .core.evaluator import MetricEvaluator

from .selection.quality import QualityThresholdSelector
from .io.loaders import load_jsonl
from .analysis.dataset_analyzer import DatasetAnalyzer
from .pipeline import InstructionTuningPipeline

# Providers
llm = OpenAILLMProvider()
embedder = OpenAIEmbeddingProvider()  # future semantic metrics

# Metrics
metrics = [
    ComplexityMetric(),
    InstructionFollowingDifficultyMetric(),
    LLMQualityMetric(llm),
    # Add more metrics here
]

weights = {
    "complexity": 0.3,
    "ifd": -0.2,        # inverted difficulty
    "quality": 0.9,
}

evaluator = MetricEvaluator(metrics, weights)
selector = QualityThresholdSelector(evaluator, threshold=0.6)
analyzer = DatasetAnalyzer(evaluator)

pipeline = InstructionTuningPipeline(evaluator, selector, analyzer)

dataset = load_jsonl("data/sft/data_sft.jsonl")
result = pipeline.run(dataset)

print(result["original"], "→", result["selected"])
