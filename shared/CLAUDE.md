# Shared Layer Rules

## This Directory
Cross-pipeline shared utilities. Everything here is GENERIC — no pipeline-specific logic.
If it only makes sense for one pipeline, it belongs in that pipeline.

## Key Files

| File | Contents |
|------|----------|
| `models.py` | Canonical data models for ALL pipelines |
| `config.py` | Pydantic v2 config schemas for all pipelines |
| `providers.py` | `BaseLLM` (re-exported from agentsoul), `SyncLLMAdapter` |
| `provider_factory.py` | `create_llm(config)` — routes to OpenAI or HuggingFace |
| `registry.py` | Generic `Registry` with decorator; `generator_registry`, `metric_registry`, `exporter_registry` |
| `converters.py` | Format conversion: SFT ↔ DPO ↔ GRPO ↔ KTO |
| `exceptions.py` | Service-level exception hierarchy (create in rebuild) |

## Canonical Data Models (`models.py`)
All pipelines must use these — never define competing data types elsewhere:
- `BaseDataPoint(instruction, input, output, metadata)` — universal SFT format
- `DPODataPoint(prompt, chosen, rejected)`
- `GRPODataPoint(prompt, responses: List[str], rewards: List[float])`
- `KTODataPoint(prompt, completion, label: bool)`

## Config Schemas (`config.py`)
All are Pydantic v2 models with `model_config`:
`PipelineConfig` → `GeneratorConfig`, `EvaluatorConfig`, `CleaningConfig`,
`NormalizationConfig`, `FinetuningConfig`, `HostingConfig`, `OrchestrationConfig`

## SyncLLMAdapter
Wraps an async `BaseLLM` for synchronous calls (used by evaluator metrics that
can't be async). Handles nested event loop detection automatically. Use when a
metric's `compute()` method must call an LLM synchronously.

## Exception Hierarchy (to create in rebuild)
```
AgentYError (base)
├── PipelineError
│   ├── GenerationError
│   ├── EvaluationError
│   ├── TrainingError
│   └── HostingError
├── ValidationError
└── ProviderError
```
Services raise typed exceptions; MCP servers catch and return error responses.

## Registry Pattern
```python
from shared.registry import generator_registry

@generator_registry.register("my_technique")
class MyGenerator(BaseGenerator):
    ...
```
