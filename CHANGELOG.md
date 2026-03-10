# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- Dataset persistence layer: `dataset.save`, `dataset.load`, `dataset.preview`, `dataset.info`, `dataset.split`, `dataset.merge` (6 tools)
- `generate.from_text` and `generate.from_hf_dataset` tools for text and HuggingFace Hub data
- `system.setup_check` and `system.config` tools for environment validation
- `finetune.merge_adapter` (LoRA merge) and `finetune.export_gguf` (GGUF export) tools
- `host.health` tool for deployment health checks
- Adapter service (`finetuning_pipeline/services/adapter_service.py`) for merge and GGUF export
- 6 standalone MCP servers: `mcp-tuna-data`, `mcp-tuna-eval`, `mcp-tuna-model-eval`, `mcp-tuna-train`, `mcp-tuna-host`, `mcp-tuna-orchestrate`
- Per-server optional dependency groups in `pyproject.toml` (install only what you need)
- `argparse` CLI for gateway and all split servers (`--help`, `--version`, `--port`)
- Graceful `ImportError` messages when optional extras are missing
- MCP Registry metadata (`server.json`)
- Example MCP client configs for Claude Desktop, Cursor, and HTTP mode
- Dockerfile (GPU + CPU targets) and `.dockerignore`
- `mcp-tuna-gateway` service in `docker-compose.yml` with GPU reservation
- CI/CD workflows: lint/test (Python 3.11 + 3.12 matrix), PyPI publish (OIDC), Docker build
- PyPI classifiers, `CHANGELOG.md`, `LICENSE` (MIT), `CONTRIBUTING.md`
- Ruff `per-file-ignores` config for intentional re-exports and dotenv patterns

### Changed
- Moved `JsonExtractor` and `DatasetExporter` to `shared/` (fixes cross-pipeline import violation)
- `data_generator_pipeline` originals now re-export from `shared/` for backward compatibility
- `EvalServer` now uses lazy service initialization (consistent with all other servers)
- Version bumped to 0.2.0 in `pyproject.toml` and server CLI `--version`

### Fixed
- `orchestration/orchestration_trainer.py` no longer imports from `data_generator_pipeline` directly
- Resolved 225 lint errors across the entire codebase (unused imports, redefined classes, unused variables)
- Restored `src/agentsoul/py.typed` marker file
- Corrected README server tool counts to match actual registrations

## [0.1.0] - 2025-05-01

### Added
- Initial release: 82 MCP tools across 11 namespaces
- Unified gateway (`mcp_gateway.py`) + 6 split servers
- Data generation (SFT/DPO/GRPO), cleaning, normalization, evaluation pipelines
- LoRA fine-tuning with 4-bit quantization
- Model hosting as MCP tool or FastAPI endpoint
- Schema-aware orchestration training data generation
