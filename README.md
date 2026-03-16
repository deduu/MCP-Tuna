# MCP Tuna

<!-- mcp-name: io.github.deduu/mcp-tuna -->

<p align="center">
  <img src="./docs/assets/mcp-tuna-logo.svg?raw=1" alt="MCP Tuna" width="760">
</p>

<p align="center">
  <strong>MCP-native platform for dataset generation, evaluation, fine-tuning, deployment, and orchestration.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#unified-gateway-tool-surface">Tool Surface</a> |
  <a href="docs/tool-catalog.md">Tool Catalog</a> |
  <a href="#frontend-surface">Frontend</a> |
  <a href="#mcp-client-setup">MCP Client Setup</a>
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-2563EB?style=flat-square">
  <img alt="MCP Gateway" src="https://img.shields.io/badge/MCP-unified_gateway-0891B2?style=flat-square">
  <img alt="Frontend" src="https://img.shields.io/badge/frontend-React%20%2B%20TypeScript-0F172A?style=flat-square">
  <img alt="Training" src="https://img.shields.io/badge/training-LoRA%20%7C%20SFT%20%7C%20DPO%20%7C%20GRPO-1D4ED8?style=flat-square">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square">
</p>

MCP Tuna is an end-to-end LLM fine-tuning platform built around a unified MCP gateway. It covers document ingestion, synthetic dataset generation, cleaning, normalization, quality scoring, LoRA training, model evaluation, deployment, and orchestration training data in one repo.

It is designed to be useful in two modes:

- as a serious MCP server suite that coding agents can call directly
- as a developer-facing control plane with a frontend for tools, datasets, training, deployments, and evaluation

## Who This Is For

- teams building fine-tuning and post-training workflows around local or hosted models
- developers who want one repo for generation, evaluation, training, deployment, and MCP integration
- AI agent workflows that need a broad, tool-rich gateway instead of one-off scripts

## Why MCP Tuna Instead Of An Ad-Hoc Stack

- You do not need to stitch together separate repos for generation, quality filtering, fine-tuning, deployment, and evaluation.
- The MCP gateway gives agents a single discovery surface instead of fragile custom glue code.
- The frontend makes the system inspectable for humans, which matters when debugging datasets, jobs, and deployments.
- The repo supports both granular tools and higher-level workflows, so it works for experimentation and production-like flows.
- It is structured as a platform, not a notebook dump or a thin model wrapper.

## Why This Repo Is Worth Trying

- It covers the full post-training lifecycle, not just fine-tuning.
- The unified gateway exposes 100+ tools across 17 namespaces through one MCP endpoint.
- The frontend is not a demo shell. It includes tool exploration, dataset flows, training controls, deployment management, and an evaluation hub.
- The repo is modular. You can run the full gateway or split servers by capability.
- It is practical for both local experimentation and agent-driven workflows through Codex, Claude Desktop, Cursor, Windsurf, or other MCP clients.

## What You Get

| Area | What it does |
|------|---------------|
| `extract` + `generate` | Load documents and create SFT, DPO, GRPO, or KTO-ready datasets |
| `clean` + `normalize` + `evaluate` | Remove bad rows, standardize schema, and score/filter quality |
| `dataset` | Save, load, preview, split, merge, and inspect datasets |
| `finetune` | Run synchronous or async LoRA training, curriculum training, and sequential multi-stage training |
| `test` + `evaluate_model` + `ft_eval` + `judge` | Compare models, benchmark runs, evaluate fine-tuned models, and run LLM-as-a-judge scoring |
| `host` | Deploy models as MCP or API services and chat against them |
| `workflow` + `orchestration` | Run end-to-end pipelines or build orchestration training data from trajectories |
| `system` + `validate` + `file` | Resource checks, model discovery, path browsing, uploads, and runtime setup |

## Unified Gateway Tool Surface

The unified gateway is the main product surface. It groups tools by namespace so the system stays discoverable instead of turning into a flat list.

| Namespace | Purpose | Typical use |
|-----------|---------|-------------|
| `system` | Resource checks, setup validation, config inspection, environment helpers | Check whether a machine can train before starting work |
| `file` | Safe file upload and browse helpers for the frontend and MCP clients | Select backend-visible files and folders |
| `extract` | Read source documents into text | Load PDFs, Markdown, text, and similar source material |
| `generate` | Create synthetic training data | Turn docs or raw text into SFT/DPO/GRPO/KTO examples |
| `clean` | Remove duplicates and invalid rows | Clean generated or imported data before evaluation |
| `normalize` | Align keys and formats | Convert mixed datasets into a target training schema |
| `evaluate` | Score dataset quality and filter low-value rows | Build higher-quality training corpora |
| `evaluate_model` | Benchmark models on structured test sets | Measure model quality and export results |
| `finetune` | Run training jobs and related utilities | Train, monitor, merge adapters, export GGUF |
| `test` | Inference and direct model comparisons | Compare base vs fine-tuned behavior |
| `validate` | Model discovery and dataset validation | Inspect model paths, search models, validate dataset schema |
| `host` | Serve trained models | Deploy MCP or API inference endpoints |
| `workflow` | Chain multiple tools into larger pipelines | Run full generate-to-train or compare workflows |
| `orchestration` | Create orchestration training datasets | Generate problems, collect trajectories, build data |
| `judge` | LLM-as-a-judge scoring | Single, batch, rubric, and pairwise evaluation |
| `ft_eval` | Domain-specific fine-tune evaluation | Evaluate generated vs reference answers with summary/export |
| `dataset` | Dataset persistence and transformation | Save, load, split, merge, inspect datasets |

For the namespace-by-namespace catalog and representative tool usage, see [docs/tool-catalog.md](docs/tool-catalog.md).

## Frontend Surface

The frontend is a working control plane, not a placeholder.

| Page | What it is for |
|------|-----------------|
| Dashboard | System status and quick navigation |
| Chat | OpenAI-compatible chat flow backed by the repo runtime |
| Tools | Live schema-driven tool explorer for every MCP tool |
| Pipeline | Guided orchestration and custom pipeline forms |
| Datasets | Import, generate, inspect, split, merge, clean, and evaluate datasets |
| Training | Configure and monitor training jobs |
| Deployments | Launch and manage hosted models |
| Evaluation | LLM Judge, Fine-tune Eval, and Model Benchmark workflows |

The Tools page is especially useful for discovery because it renders tool parameters from the MCP schema and now supports dropdowns for constrained inputs plus browse controls for path-like fields.

## UI Preview

<p align="center">
  <img src="docs/assets/screenshots/platform-gallery.svg" alt="MCP Tuna UI preview gallery" width="1080">
</p>

<p align="center">
  Visual preview of the core operator surfaces: Tools, Training, Evaluation, and Deployments.
</p>

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/deduu/MCP-Tuna.git
cd MCP-Tuna
uv sync --extra all
```

### 2. Run the unified gateway for MCP clients

```bash
uv run mcp-tuna-gateway http --port 8000
```

For stdio-based MCP clients:

```bash
uv run mcp-tuna-gateway
```

### 3. Run the local frontend stack

The frontend dev proxy expects:

- FastAPI backend on `http://127.0.0.1:8000`
- MCP gateway on `http://127.0.0.1:8002`

Start them in separate terminals:

```bash
uv run uvicorn app.main:app --reload --port 8000
uv run python scripts/run_gateway.py http --port 8002
```

Then run the frontend:

```bash
cd frontend
npm install
npm run dev
```

## Copy-Paste Examples

These examples use the unified gateway over HTTP.

If you are following the frontend dev setup from this README, use port `8002` for `/mcp`. If you are running the gateway standalone, `8000` is fine.

### Check whether a machine can train

```bash
curl -s http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"check-resources",
    "method":"tools/call",
    "params":{"name":"system.check_resources","arguments":{}}
  }'
```

### Generate SFT data from a document

```bash
curl -s http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"generate-doc",
    "method":"tools/call",
    "params":{
      "name":"generate.from_document",
      "arguments":{
        "technique":"sft",
        "file_path":"uploads/example.md"
      }
    }
  }'
```

### Validate a dataset before training

```bash
curl -s http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"validate-dataset",
    "method":"tools/call",
    "params":{
      "name":"validate.schema",
      "arguments":{
        "dataset_path":"output/my_dataset.jsonl",
        "technique":"sft"
      }
    }
  }'
```

### Start async fine-tuning

```bash
curl -s http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"train-async",
    "method":"tools/call",
    "params":{
      "name":"finetune.train_async",
      "arguments":{
        "dataset_path":"output/my_dataset.jsonl",
        "output_dir":"output/my_run",
        "base_model":"meta-llama/Llama-3.2-3B-Instruct"
      }
    }
  }'
```

### Run an end-to-end pipeline from a high-level goal

```bash
curl -s http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":"guided-pipeline",
    "method":"tools/call",
    "params":{
      "name":"workflow.guided_pipeline",
      "arguments":{
        "goal":"Generate a clean SFT dataset from a markdown document and prepare it for fine-tuning",
        "file_path":"uploads/example.md",
        "base_model":"meta-llama/Llama-3.2-3B-Instruct"
      }
    }
  }'
```

## Typical Workflows

### Build a training dataset from documents

1. Use `extract.load_document` or `generate.from_document`.
2. Clean with `clean.dataset`.
3. Normalize with `normalize.dataset`.
4. Score and filter with `evaluate.dataset` and `evaluate.filter_by_quality`.
5. Persist with `dataset.save`.

### Validate whether a machine can train

1. Run `system.check_resources`.
2. Run `system.preflight_check` for a known model.
3. Run `system.prescribe` or `system.auto_prescribe` to generate recommended settings.

### Train and deploy a model

1. Start with a validated dataset path.
2. Run `finetune.train_async` or one of the technique-specific training tools.
3. Monitor with `finetune.job_status` or `finetune.list_jobs`.
4. Deploy with `host.deploy_mcp` or `host.deploy_api`.

### Benchmark multiple models

1. Prepare a test dataset.
2. Use the Evaluation Hub `Model Benchmark` page or call `evaluate_model.batch`.
3. Export results with `evaluate_model.export`.

## Installation Tiers

Install only what you need:

| Extra | What it enables | Key dependencies |
|-------|------------------|------------------|
| `data` | Dataset generation | openai |
| `eval` | Dataset quality scoring and filtering | openai, scikit-learn |
| `model-eval` | Model comparison and benchmarking | rouge-score, evaluate, pandas |
| `model-eval-full` | `model-eval` plus BERTScore | bert-score |
| `training` | LoRA fine-tuning | torch, transformers, peft, trl |
| `hosting` | Model deployment | torch, transformers, fastapi |
| `orchestration` | Agent trajectory training data | openai |
| `export` | GGUF export | llama-cpp-python |
| `backend` | FastAPI backend and PostgreSQL | sqlalchemy, asyncpg |
| `memory` | Agent memory integrations | chromadb |
| `retrieval` | BM25 and FAISS retrieval | faiss-cpu, rank-bm25 |
| `tracing` | Observability | auditi |
| `dev` | Tests and linting | pytest, ruff |
| `all-servers` | All MCP server extras | combined server extras |
| `all` | Full local development environment | all server + infra extras |

Examples:

```bash
pip install "mcp-tuna[data]"
pip install "mcp-tuna[training,hosting]"
pip install "mcp-tuna[all]"
```

## Available Server Commands

| Server | Command | Primary use |
|--------|---------|-------------|
| Unified Gateway | `mcp-tuna-gateway` | Full end-to-end MCP surface |
| Data Prep | `mcp-tuna-data` | Document loading, generation, cleaning, normalization, dataset IO |
| Evaluation | `mcp-tuna-eval` | Dataset scoring and filtering |
| Model Eval | `mcp-tuna-model-eval` | Benchmarking and judge utilities |
| Training | `mcp-tuna-train` | Fine-tuning workflows |
| Hosting | `mcp-tuna-host` | Deployment and health checks |
| Orchestration | `mcp-tuna-orchestrate` | Trajectory generation and orchestration datasets |
| Chat | `mcp-tuna-chat` | Direct model chat CLI |

## MCP Client Setup

### Auto-generate configs

```bash
mcp-tuna-setup --all
```

Zero-install with `uvx`:

```bash
mcp-tuna-setup --all --launcher uvx
```

### Codex

Add to `~/.codex/config.toml` or `.codex/config.toml`:

```toml
[mcp_servers.mcp-tuna]
command = "mcp-tuna-gateway"

[mcp_servers.mcp-tuna.env]
OPENAI_API_KEY = "sk-..."
HF_TOKEN = "hf_..."
```

HTTP mode:

```toml
[mcp_servers.mcp-tuna]
url = "http://localhost:8000/mcp"
```

### Claude Desktop

```json
{
  "mcpServers": {
    "mcp-tuna": {
      "command": "mcp-tuna-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### Cursor

```json
{
  "mcpServers": {
    "mcp-tuna": {
      "command": "mcp-tuna-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

More ready-to-use configs live in [examples](examples).

## Docker

```bash
# GPU image
docker build -t mcp-tuna .
docker run --gpus all -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v hf-cache:/root/.cache/huggingface \
  mcp-tuna

# CPU-focused image
docker build --target cpu -t mcp-tuna-cpu .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... mcp-tuna-cpu

# docker compose
docker compose up -d
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For generation and evaluation flows | OpenAI API access |
| `HF_TOKEN` | For training or push-to-hub flows | Hugging Face downloads and uploads |
| `HF_HOME` | No | Hugging Face cache location |
| `MODEL_ROOT` | No | Extra server-visible model folder for the frontend browser |
| `MODEL_BROWSE_ROOTS` | No | Multiple extra model folders, separated by OS path separator |

## Architecture

```text
Documents / Raw Text / HF Datasets
                |
                v
      Extract + Generate Pipelines
                |
      +---------+---------+---------+
      |         |         |         |
      v         v         v         v
    Clean   Normalize  Evaluate  Dataset IO
      \         |         /         /
       +--------+--------+---------+
                |
                v
         Fine-tuning Pipeline
                |
       +--------+---------+
       |                  |
       v                  v
   Evaluation         Hosting
                |
                v
         Orchestration Data
```

## Project Structure

```text
mcp-tuna/
|-- mcp_gateway.py               # Unified MCP gateway
|-- app/                         # FastAPI backend and runtime helpers
|-- frontend/                    # React control plane
|-- scripts/                     # Entry points, setup, and e2e helpers
|-- servers/                     # Split MCP servers
|-- shared/                      # Shared config, models, and utilities
|-- data_generator_pipeline/     # Dataset generation
|-- data_cleaning_pipeline/      # Deduplication and schema validation
|-- data_normalization_pipeline/ # Key and format normalization
|-- data_evaluator_pipeline/     # Dataset scoring and filtering
|-- model_evaluator_pipeline/    # Benchmarking, judge, and FT eval
|-- finetuning_pipeline/         # LoRA training and inference
|-- hosting_pipeline/            # Deployment services
|-- orchestration/               # Orchestration training data workflows
`-- examples/                    # MCP client config examples
```

## Development

```bash
# full local environment
uv sync --extra all

# tests
uv run pytest -x -q

# lint
uv run ruff check .

# frontend build
cd frontend
npm install
npm run build
```

## License

MIT
