# Data Evaluator Pipeline Rules

## This Directory
Scores datasets on 4 quality dimensions. Used to filter low-quality training data
before fine-tuning. Service API exposed via `services/pipeline_service.py`.

## Structure
```
data_evaluator_pipeline/
├── core/
│   ├── data.py          # DataPoint dataclass (internal to this pipeline)
│   ├── evaluator.py     # MetricEvaluator — applies weighted scoring
│   └── metrics/
│       ├── base.py      # BaseMetric ABC
│       ├── complexity.py   # Vocabulary richness + semantic density
│       ├── ifd.py          # Instruction-Following Difficulty
│       ├── quality.py      # LLM-judged quality (calls OpenAI)
│       ├── similarity.py   # Cosine similarity (embeddings)
│       └── configs/        # Language configs: en.json, id.json
├── providers/           # OpenAIProvider, CohereEmbedding, OpenAIEmbedding
├── selection/quality.py # QualityThresholdSelector
├── analysis/            # Dataset statistics (mean, std, min, max per metric)
├── io/loaders.py        # load_jsonl()
├── services/pipeline_service.py  # Stateless service API
└── pipeline.py          # InstructionTuningPipeline (high-level orchestrator)
```

## Scoring Formula
`final_score = Σ(weight_i × score_i)` where weights sum to 1.0.
Default weights in `EvaluatorConfig` (overridable): complexity, IFD, quality, similarity.

## SyncLLMAdapter
`quality.py` metric needs to call LLMs synchronously. Use `SyncLLMAdapter` from
`shared/providers.py` — it handles nested event loops safely.

## Service API (`services/pipeline_service.py`)
- `evaluate_dataset(data, config)` → scored dataset
- `filter_by_quality(data, min_score)` → filtered dataset
- `analyze_statistics(data)` → per-metric stats dict
- `list_metrics()` → registered metric names

## Adding a New Metric
1. Write tests in `tests/data_evaluator_pipeline/test_<name>.py` FIRST
2. Create `core/metrics/<name>.py` inheriting `BaseMetric`
3. Implement `compute(datapoint: DataPoint) -> float` — return 0.0–1.0
4. Add language config in `core/metrics/configs/` if needed
5. Add to `MetricEvaluator` metric list and `EvaluatorConfig` weight key

## Rules
- Metrics MUST return float in [0.0, 1.0] — normalize before returning
- Metrics that call LLMs must use `SyncLLMAdapter` (not raw async in sync context)
- `QualityThresholdSelector` is the ONLY way to filter — never filter in service layer
