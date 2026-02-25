# Data Generator Pipeline Rules

## This Directory
Generates SFT/DPO/GRPO training datasets from documents (PDF, MD, DOCX).
Service API is exposed via `services/pipeline_service.py` → wrapped by `mcp/server.py`.

## Structure
```
data_generator_pipeline/
├── core/
│   ├── base.py          # BaseGenerator, BaseParser, BaseLLM ABCs
│   ├── factory.py       # PipelineFactory.create(technique, llm, config)
│   └── pipeline.py      # FinetuningPipeline orchestrator
├── generators/
│   ├── sft.py, dpo.py, grpo.py    # Technique implementations
│   └── registry.py      # @generator_registry.register("name")
├── models/datapoints.py # BaseDataPoint, SFTDataPoint (with source tracking)
├── exporters/dataset.py # JSON, JSONL, CSV, Excel, HuggingFace export
├── parsers/             # JSON extraction from LLM responses
├── prompts/templates.py # PromptTemplateManager
├── services/pipeline_service.py   # Stateless service API (MCP-facing)
└── mcp/server.py        # MCP tool definitions
```

## Key Patterns
- `PipelineFactory.create(technique, llm, config)` — builds the correct pipeline for a technique
- `SFTDataPoint` extends `BaseDataPoint` with source tracking: `id`, `file_name`, `page`, `text`
- `PromptTemplateManager` loads templates from `prompts/` — never hardcode prompts in generators
- `GeneratorRegistry` wraps `shared.registry.generator_registry` for backward compat

## Service API (`services/pipeline_service.py`)
All methods are async:
- `load_document(file_path)` — PDF/DOCX → list of pages
- `generate_from_page(technique, page_text, ...)` → list of datapoints
- `generate_from_document(technique, file_path, page_range, ...)` → full dataset
- `generate_batch(technique, file_paths, ...)` → multi-doc dataset
- `export_dataset(data, format, output_path)` — JSON/JSONL/CSV/Excel/HuggingFace
- `list_techniques()` → registered generator names

## Adding a New Technique
1. Write tests in `tests/data_generator_pipeline/test_<name>.py` FIRST
2. Create `generators/<name>.py` inheriting `BaseGenerator`
3. Register: `@generator_registry.register("<name>")`
4. Add prompt template in `prompts/`
5. Export from `generators/__init__.py`

## Rules
- Generators NEVER call LLMs directly — always go through `BaseLLM` interface
- Source tracking is mandatory in all datapoint models (file_name, page)
- Exporters NEVER contain business logic — only format conversion
