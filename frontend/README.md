# MCP Tuna Frontend

React + TypeScript control plane for MCP Tuna.

This frontend is the operator UI for the local platform. It is not a generic Vite starter. It provides working pages for:

- tool exploration across the full MCP gateway
- dataset import, generation, cleaning, split, merge, and evaluation
- training configuration and job monitoring
- deployment launch and inspection
- evaluation workflows including LLM Judge, Fine-tune Eval, and Model Benchmark

## Dev Setup

The Vite dev server expects these local backends:

- `/v1` -> `http://127.0.0.1:8000`
- `/mcp` -> `http://127.0.0.1:8002`

Start them first from the repo root:

```bash
uv run uvicorn app.main:app --reload --port 8000
uv run python scripts/run_gateway.py http --port 8002
```

Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Notes

- Tool forms are schema-driven from MCP `tools/list`.
- Path selection uses backend-visible browse roots exposed by the gateway.
- Model selection supports both local backend paths and Hugging Face model IDs.
