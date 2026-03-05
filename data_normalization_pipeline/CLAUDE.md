# Data Normalization Pipeline Rules

## This Directory
Format conversion and key standardization for training datasets (SFT ↔ DPO ↔ GRPO ↔ KTO).
Service API is exposed via `services/normalization_service.py` → wrapped by `mcp/server.py`.

## Structure
```
data_normalization_pipeline/
├── services/
│   └── normalization_service.py   # DataNormalizationService — stateless async API
└── mcp/
    └── server.py                  # NormalizationMCPServer — MCP tool definitions
```

## Service API (`services/normalization_service.py`)
All methods are async, operate on `List[Dict[str, Any]]`:
- `normalize_dataset(data_points, config?)` — runs all normalization steps in sequence
- `merge_instruction_input(data_points)` — combines `instruction` + `input` into single field
- `standardize_keys(data_points, target_format)` — remaps alternate field names to canonical keys
- `strip_and_clean_text(data_points)` — whitespace trimming + Unicode NFC normalization

## Key Remapping
Maps alternate field names to canonical ones:
- `prompt` / `question` / `query` → `instruction`
- `response` / `answer` / `completion` → `output`
- `context` → `input`

Target formats: `sft` (instruction/input/output), `dpo` (prompt/chosen/rejected), `grpo` (prompt/responses/rewards)

## Config (`shared/config.py` → `NormalizationConfig`)
```python
NormalizationConfig(
    target_format="sft",
    merge_instruction_input=True,
    strip_whitespace=True,
)
```

## MCP Tools Exposed
`normalize.dataset`, `normalize.merge_fields`, `normalize.standardize_keys`, `normalize.strip_text`

## Rules
- Key naming follows `shared/models.py` canonical fields — never invent new field names
- Unicode normalization uses `unicodedata.normalize("NFC", text)` — always NFC, never NFD
- Imports from `shared/` only — never import other pipelines
