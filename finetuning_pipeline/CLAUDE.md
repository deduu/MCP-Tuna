# Finetuning Pipeline Rules

## This Directory
LoRA fine-tuning using PEFT + trl. Uses a facade pattern: `pipeline_service.py`
composes 4 focused sub-services. All heavy GPU operations are isolated in sub-services.

## Structure
```
finetuning_pipeline/
├── services/
│   ├── pipeline_service.py      # Facade — composes all 4 sub-services
│   ├── gpu_service.py           # GPU memory management (torch)
│   ├── training_service.py      # LoRA training with trl SFTTrainer
│   ├── inference_service.py     # Run inference + compare models
│   └── model_discovery_service.py  # HuggingFace search + local model scan
├── scripts/run_mcp_server.py    # MCP server launcher
└── tests/unit/test.py           # Unit tests (TDD required)
```

## Facade Pattern
`FineTuningService` in `pipeline_service.py` lazy-initializes sub-services on first call:
- `GPUService.clear_gpu_memory()` — frees VRAM before/after training
- `TrainingService.train_model(dataset, config)` — LoRA via PEFT + trl
- `InferenceService.run_inference(model_path, inputs)` — generation
- `InferenceService.compare_models(base, finetuned, prompts)` — side-by-side
- `ModelDiscoveryService.list_available_base_models()` — local + HF
- `ModelDiscoveryService.search_huggingface_models(query)` — HF Hub search
- `ModelDiscoveryService.get_recommended_models()` — curated list

## Config (`shared/config.py` → `FinetuningConfig`)
```python
FinetuningConfig(
    base_model="Qwen/Qwen3-1.7B",
    num_epochs=3,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
)
```

## GPU Setup
Always set `PYTORCH_ALLOC_CONF=expandable_segments:True` in environment.
Call `GPUService.clear_gpu_memory()` before training and after inference.

## MCP Tools Exposed
`finetune.train`, `finetune.infer`, `finetune.compare`, `validate.list_models`,
`validate.search_models`, `validate.recommend_models`, `validate.get_model_info`

## Rules
- Training NEVER runs in the request thread — always in a background task or separate process
- `pipeline_service.py` is the ONLY public API; sub-services are internal
- Tests mock GPU operations — never run real torch training in unit tests
