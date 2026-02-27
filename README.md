# Transcendence

End-to-end LLM fine-tuning platform that generates, evaluates, and trains on custom datasets. Exposed as **82 MCP tools** across 11 namespaces — first-of-its-kind for the MCP ecosystem.

## Quick Start

```bash
# Install (all servers)
pip install transcendence[all-servers]

# Or with uv
uv pip install transcendence[all-servers]
```

Add to your MCP client and start fine-tuning with natural language:

```
"Generate an SFT dataset from my docs, evaluate quality, train a LoRA adapter, and deploy it."
```

## Installation Tiers

Install only what you need:

| Extra | What it enables | Key dependencies |
|-------|----------------|------------------|
| `data` | Dataset generation (SFT/DPO/GRPO) | openai |
| `eval` | Dataset quality scoring & filtering | openai, scikit-learn |
| `model-eval` | Model comparison & benchmarking | rouge-score, evaluate, pandas |
| `model-eval-full` | + BERTScore metric | bert-score |
| `training` | LoRA fine-tuning | torch, transformers, peft, trl |
| `hosting` | Model deployment | torch, transformers, fastapi |
| `orchestration` | Agent trajectory training data | openai |
| `all-servers` | Everything above | all |

```bash
# Data generation only (no GPU needed)
pip install transcendence[data]

# Training + hosting (GPU)
pip install transcendence[training,hosting]
```

## Available Servers

| Server | Command | Tools | Required extra |
|--------|---------|-------|----------------|
| **Unified Gateway** | `transcendence-gateway` | 82 | `all-servers` |
| Data Prep | `transcendence-data` | 22 | `data` |
| Evaluation | `transcendence-eval` | 4 | `eval` |
| Model Eval | `transcendence-model-eval` | 15 | `model-eval` |
| Training | `transcendence-train` | 11 | `training` |
| Hosting | `transcendence-host` | 5 | `hosting` |
| Orchestration | `transcendence-orchestrate` | 3 | `orchestration` |

All servers support two transport modes:

```bash
transcendence-gateway              # stdio mode (Claude Desktop, Cursor, Claude Code)
transcendence-gateway http         # HTTP mode (default port 8000)
transcendence-gateway http --port 9000
transcendence-gateway --version
transcendence-gateway --help
```

## MCP Client Setup

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "transcendence": {
      "command": "transcendence-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

Or zero-install with `uvx` (no `pip install` needed):

```json
{
  "mcpServers": {
    "transcendence": {
      "command": "uvx",
      "args": ["--from", "transcendence[all-servers]", "transcendence-gateway"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### Codex (OpenAI)

Add to `~/.codex/config.toml` (global) or `.codex/config.toml` (project-scoped):

**Stdio mode** (Codex launches the server automatically):

```toml
[mcp_servers.transcendence]
command = "transcendence-gateway"

[mcp_servers.transcendence.env]
OPENAI_API_KEY = "sk-..."
HF_TOKEN = "hf_..."
```

**HTTP mode** (connect to a running server):

```toml
[mcp_servers.transcendence]
url = "http://localhost:8000/mcp"
```

> Codex supports both stdio and HTTP transports. The CLI and VS Code extension share the same config.

### Cursor

Place in `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "transcendence": {
      "command": "transcendence-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### Windsurf

Edit `~/.codeium/windsurf/mcp_config.json` (or add via Windsurf Settings > Cascade > MCP Servers):

```json
{
  "mcpServers": {
    "transcendence": {
      "command": "transcendence-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### VS Code (Copilot)

Place in `.vscode/mcp.json`:

```json
{
  "servers": {
    "transcendence": {
      "command": "transcendence-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### JetBrains (IntelliJ, PyCharm, WebStorm)

Go to **Settings > Tools > AI Assistant > Model Context Protocol (MCP)**, click **Add**, switch to **As JSON**, and paste:

```json
{
  "command": "transcendence-gateway",
  "env": { "OPENAI_API_KEY": "sk-..." }
}
```

Or click **Import from Claude** if you already have Claude Desktop configured.

### Claude Code

Place in `.mcp.json` at project root:

```json
{
  "mcpServers": {
    "transcendence": {
      "command": "transcendence-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### Split servers (lightweight, per-task)

Use individual servers if you only need a subset of tools:

```json
{
  "mcpServers": {
    "transcendence-data": {
      "command": "transcendence-data",
      "env": { "OPENAI_API_KEY": "sk-..." }
    },
    "transcendence-train": {
      "command": "transcendence-train",
      "env": { "HF_TOKEN": "hf_..." }
    }
  }
}
```

See `examples/` for more configuration templates.

## Docker

```bash
# GPU (all servers, CUDA 12.4)
docker build -t transcendence .
docker run --gpus all -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v hf-cache:/root/.cache/huggingface \
  transcendence

# CPU (data + eval only, no torch)
docker build --target cpu -t transcendence-cpu .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... transcendence-cpu

# docker compose (includes PostgreSQL)
docker compose up -d
```

Then connect any MCP client via HTTP:

```toml
# Codex example
[mcp_servers.transcendence]
url = "http://localhost:8000/mcp"
```

## Architecture

```
Documents (PDF, MD, Excel)
                |
                v
  ┌──────────────────────────────┐
  │  Data Generator              │  Generate SFT / DPO / GRPO datasets from documents
  │  Pipeline                    │  using LLMs (OpenAI, custom BaseLLM)
  └──────────────┬───────────────┘
                 |
       ┌─────────┼─────────┐
       v         v         v
  ┌─────────┐ ┌────────┐ ┌──────────┐
  │ Cleaner │ │Normaliz│ │Evaluator │  Clean, normalize, score & filter
  └────┬────┘ └───┬────┘ └────┬─────┘
       └──────────┼───────────┘
                  v
  ┌──────────────────────────────┐
  │  Fine-tuning                 │  Train LoRA adapters, run inference,
  │  Pipeline                    │  compare base vs fine-tuned models
  └──────────────┬───────────────┘
                 v
  ┌──────────────────────────────┐
  │  Hosting                     │  Deploy as MCP tool or FastAPI endpoint
  │  Pipeline                    │
  └──────────────────────────────┘
```

**Orchestration path:** Generate problems → Collect agent trajectories → Score (accuracy + cost + latency) → Format → Train → Host

## Development

```bash
# Install all dependencies
uv sync --all-extras

# Run tests
uv run pytest -x -q

# Lint & format
uv run ruff check .
uv run ruff format .

# Start gateway locally
python scripts/run_gateway.py http 8000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For data/eval | OpenAI API access |
| `HF_TOKEN` | For training | HuggingFace model downloads and hub push |
| `HF_HOME` | No | HuggingFace model cache directory |

## Project Structure

```
AgentY/
├── mcp_gateway.py              # Unified MCP gateway (82 tools)
├── scripts/                    # Entry points for all servers
├── servers/                    # Standalone split server implementations
├── shared/                     # Cross-pipeline models, config, utilities
├── data_generator_pipeline/    # SFT/DPO/GRPO dataset generation
├── data_cleaning_pipeline/     # Deduplication & schema validation
├── data_normalization_pipeline/# Format conversion (SFT ↔ DPO ↔ GRPO ↔ KTO)
├── data_evaluator_pipeline/    # Quality scoring & filtering
├── model_evaluator_pipeline/   # Model comparison & benchmarking
├── finetuning_pipeline/        # LoRA training + inference
├── hosting_pipeline/           # Model deployment
├── orchestration/              # Agent trajectory training data
├── src/agentsoul/              # Bundled agent framework
└── app/                        # FastAPI backend
```

## License

MIT
