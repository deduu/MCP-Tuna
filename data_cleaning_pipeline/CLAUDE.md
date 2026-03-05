# Data Cleaning Pipeline Rules

## This Directory
Deduplication, schema validation, and quality filtering for training datasets.
Service API is exposed via `services/cleaning_service.py` → wrapped by `mcp/server.py`.

## Structure
```
data_cleaning_pipeline/
├── services/
│   └── cleaning_service.py    # DataCleaningService — stateless async API
└── mcp/
    └── server.py              # CleaningMCPServer — MCP tool definitions
```

## Service API (`services/cleaning_service.py`)
All methods are async, operate on `List[Dict[str, Any]]`:
- `clean_dataset(data_points, config?)` — runs all cleaning steps: empty fields, duplicates, short entries
- `deduplicate(data_points, key="instruction")` — removes exact-match duplicates on a key
- `validate_schema(data_points, technique="sft")` — checks required fields per technique
- `remove_short_entries(data_points, min_instruction, min_output)` — filters by length
- `remove_empty_fields(data_points)` — drops entries with blank required fields

## Config (`shared/config.py` → `CleaningConfig`)
```python
CleaningConfig(
    remove_duplicates=True,
    min_instruction_length=10,
    min_output_length=20,
    remove_empty_fields=True,
)
```

## MCP Tools Exposed
`clean.dataset`, `clean.deduplicate`, `clean.validate_schema`, `clean.remove_short`

## Rules
- Cleaning steps are composable — `clean_dataset` chains them, but each is callable standalone
- Never modify data semantics — only remove or filter, never alter content
- Return dicts always include `original_count` and `cleaned_count` for auditability
- Imports from `shared/` only — never import other pipelines
