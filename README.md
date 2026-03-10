# MCP Tuna

<!-- mcp-name: io.github.deduu/mcp-tuna -->

End-to-end LLM fine-tuning platform and MCP server suite for dataset generation, evaluation, training, and deployment. Teams can run it from a GitHub checkout, Docker, PyPI, or an MCP client config backed by the MCP registry manifest in `server.json`.

## Quick Start

Choose the install path that matches how your team works.

### PyPI

```bash
pip install "mcp-tuna[all-servers]"
mcp-tuna-gateway
```

### GitHub / manual

```bash
git clone https://github.com/deduu/MCP-Tuna.git
cd MCP-Tuna
uv sync --extra all-servers
uv run mcp-tuna-gateway
```

### Docker

```bash
docker build -t mcp-tuna .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... mcp-tuna
```

For MCP client configs, either use the examples below directly or generate them with:

```bash
mcp-tuna-setup --all
```

## Installation Tiers

Install only what you need:

| Extra | What it enables | Key dependencies |
|-------|----------------|------------------|
| `data` | Dataset generation (SFT/DPO/GRPO) | openai |
| `eval` | Dataset quality scoring and filtering | openai, scikit-learn |
| `model-eval` | Model comparison and benchmarking | rouge-score, evaluate, pandas |
| `model-eval-full` | `model-eval` plus BERTScore | bert-score |
| `training` | LoRA fine-tuning | torch, transformers, peft, trl |
| `hosting` | Model deployment | torch, transformers, fastapi |
| `orchestration` | Agent trajectory training data | openai |
| `export` | GGUF model export | llama-cpp-python |
| `backend` | FastAPI backend and PostgreSQL | sqlalchemy, asyncpg |
| `memory` | Agent memory (vector store) | chromadb |
| `retrieval` | Retrieval (BM25 and FAISS) | faiss-cpu, rank-bm25 |
| `tracing` | Observability | auditi |
| `dev` | Development tools | pytest, ruff |
| `all-servers` | All MCP server extras | combined server extras |

```bash
# Data generation only (no GPU needed)
pip install "mcp-tuna[data]"

# Training + hosting (GPU)
pip install "mcp-tuna[training,hosting]"
```

## Available Servers

| Server | Command | Primary use | Required extra |
|--------|---------|-------------|----------------|
| Unified Gateway | `mcp-tuna-gateway` | Full end-to-end MCP surface | `all-servers` |
| Data Prep | `mcp-tuna-data` | Document loading, generation, cleaning, normalization, dataset IO | `data` |
| Evaluation | `mcp-tuna-eval` | Dataset quality scoring and filtering | `eval` |
| Model Eval | `mcp-tuna-model-eval` | Model comparison, judges, and benchmark export | `model-eval` |
| Training | `mcp-tuna-train` | LoRA fine-tuning workflows | `training` |
| Hosting | `mcp-tuna-host` | MCP/API deployment and health checks | `hosting` |
| Orchestration | `mcp-tuna-orchestrate` | Trajectory generation and orchestration datasets | `orchestration` |
| Chat | `mcp-tuna-chat` | Direct model chat CLI | `hosting` |

The gateway is the default entry point for AI coding agents such as Codex, Claude Code, Cursor, Claude Desktop, Windsurf, and JetBrains AI Assistant.

All servers support two transport modes:

```bash
mcp-tuna-gateway              # stdio mode
mcp-tuna-gateway http         # HTTP mode, default port 8000
mcp-tuna-gateway http --port 9000
mcp-tuna-gateway --version
mcp-tuna-gateway --help
```

## MCP Client Setup

### Auto-generate configs

Installed package:

```bash
mcp-tuna-setup --all
```

Zero-install `uvx` launcher:

```bash
mcp-tuna-setup --all --launcher uvx
```

Repo checkout launcher for local development:

```bash
python scripts/setup_mcp.py --all --launcher repo
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

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

Or zero-install with `uvx`:

```json
{
  "mcpServers": {
    "mcp-tuna": {
      "command": "uvx",
      "args": ["--from", "mcp-tuna[all-servers]", "mcp-tuna-gateway"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### Codex (OpenAI)

Add to `~/.codex/config.toml` or `.codex/config.toml`.

Stdio mode:

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

### Cursor

Place in `.cursor/mcp.json`:

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

### Windsurf

Edit `~/.codeium/windsurf/mcp_config.json`:

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

### VS Code (Copilot)

Place in `.vscode/mcp.json`:

```json
{
  "servers": {
    "mcp-tuna": {
      "command": "mcp-tuna-gateway",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

### JetBrains

Go to Settings > Tools > AI Assistant > Model Context Protocol (MCP), click Add, switch to As JSON, and paste:

```json
{
  "command": "mcp-tuna-gateway",
  "env": { "OPENAI_API_KEY": "sk-..." }
}
```

### Claude Code

Place in `.mcp.json` at project root:

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

### Split servers

Use individual servers if you only need a subset of tools:

```json
{
  "mcpServers": {
    "mcp-tuna-data": {
      "command": "mcp-tuna-data",
      "env": { "OPENAI_API_KEY": "sk-..." }
    },
    "mcp-tuna-train": {
      "command": "mcp-tuna-train",
      "env": { "HF_TOKEN": "hf_..." }
    }
  }
}
```

See `examples/` for more configuration templates.

## MCP Registry

The repo includes `server.json`, which is the MCP registry manifest for `mcp-tuna`. Keep it aligned with the published PyPI package and the default gateway command:

```json
{
  "name": "io.github.deduu/mcp-tuna",
  "packages": [
    {
      "registry_type": "pypi",
      "identifier": "mcp-tuna",
      "runtime_hint": "uvx"
    }
  ]
}
```

## Docker

```bash
# GPU (all servers, CUDA 12.4)
docker build -t mcp-tuna .
docker run --gpus all -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v hf-cache:/root/.cache/huggingface \
  mcp-tuna

# CPU (data + eval only, no torch)
docker build --target cpu -t mcp-tuna-cpu .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... mcp-tuna-cpu

# docker compose (includes PostgreSQL)
docker compose up -d
```

Then connect any MCP client via HTTP:

```toml
[mcp_servers.mcp-tuna]
url = "http://localhost:8000/mcp"
```

## Architecture

```text
Documents (PDF, MD, Excel)
                |
                v
  Data Generator Pipeline
                |
      +---------+---------+
      |         |         |
      v         v         v
   Cleaner   Normalizer  Evaluator
      \         |         /
       +--------+--------+
                |
                v
      Fine-tuning Pipeline
                |
                v
        Hosting Pipeline
```

Orchestration path: Generate problems -> Collect agent trajectories -> Score -> Format -> Train -> Host

## Development

```bash
# Install all dependencies
uv sync --all-extras

# Run tests
uv run pytest -x -q

# Lint
uv run ruff check .

# Start gateway locally
uv run mcp-tuna-gateway http --port 8000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For data/eval | OpenAI API access |
| `HF_TOKEN` | For training | HuggingFace model downloads and hub push |
| `HF_HOME` | No | HuggingFace model cache directory |
| `MODEL_ROOT` | No | Extra server-visible model folder to expose in the frontend model browser |
| `MODEL_BROWSE_ROOTS` | No | Multiple extra model folders for the browser, separated by the OS path separator (`;` on Windows, `:` on Unix) |

## Project Structure

```text
mcp-tuna/
|-- mcp_gateway.py              # Unified MCP gateway
|-- scripts/                    # CLI entry points and setup helper
|-- servers/                    # Split server implementations
|-- shared/                     # Cross-pipeline models, config, utilities
|-- data_generator_pipeline/    # SFT/DPO/GRPO dataset generation
|-- data_cleaning_pipeline/     # Deduplication and schema validation
|-- data_normalization_pipeline/# Format conversion
|-- data_evaluator_pipeline/    # Quality scoring and filtering
|-- model_evaluator_pipeline/   # Model comparison and benchmarking
|-- finetuning_pipeline/        # LoRA training and inference
|-- hosting_pipeline/           # Model deployment
|-- orchestration/              # Agent trajectory training data
|-- src/agentsoul/              # Bundled agent framework
`-- app/                        # Gateway runtime helpers and backend code
```

## License

MIT
