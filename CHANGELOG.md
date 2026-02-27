# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- `argparse` CLI for gateway and all split servers (`--help`, `--version`, `--port`)
- Graceful `ImportError` messages when optional extras are missing
- MCP Registry metadata (`server.json`)
- Example MCP client configs for Claude Desktop, Cursor, and HTTP mode
- Dockerfile (GPU + CPU targets) and `.dockerignore`
- `transcendence-gateway` service in `docker-compose.yml` with GPU reservation
- CI/CD workflows: lint/test, PyPI publish (OIDC), Docker build
- PyPI classifiers and `CHANGELOG.md`

### Changed
- Moved `JsonExtractor` and `DatasetExporter` to `shared/` (fixes cross-pipeline import violation)
- `data_generator_pipeline` originals now re-export from `shared/` for backward compatibility

### Fixed
- `orchestration/orchestration_trainer.py` no longer imports from `data_generator_pipeline` directly

## [0.1.0] - 2025-05-01

### Added
- Initial release: 82 MCP tools across 11 namespaces
- Unified gateway (`mcp_gateway.py`) + 6 split servers
- Data generation (SFT/DPO/GRPO), cleaning, normalization, evaluation pipelines
- LoRA fine-tuning with 4-bit quantization
- Model hosting as MCP tool or FastAPI endpoint
- Schema-aware orchestration training data generation
