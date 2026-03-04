# MCP Tuna Diagnostic Logs — AI Agent Guide

This file tells you (Claude) exactly how to read session logs, write findings,
and request additional data when diagnosing bugs in the MCP Tuna backend.

---

## Directory Structure

```
logs/
├── agentic.log              # Plain-text rotating log from AgentSoul — READ ONLY, never modify
├── sessions/
│   └── <session-uuid>.jsonl # One JSONL file per backend process lifetime (one line = one event)
└── ai_diagnostics/
    ├── findings.jsonl       # YOUR findings — append using AIDiagnosticWriter
    └── data_requests.jsonl  # YOUR data requests — append using AIDiagnosticWriter
```

---

## Reading Session Logs

Each line in `logs/sessions/<uuid>.jsonl` is a JSON object with these shared fields:

| Field        | Type   | Description                                      |
|--------------|--------|--------------------------------------------------|
| `ts`         | string | ISO-8601 UTC timestamp                           |
| `trace_id`   | string | UUID per HTTP request — correlates all events    |
| `session_id` | string | UUID per backend process lifetime                |
| `event_type` | string | One of the event types listed below              |

### Event Types

| `event_type`     | When emitted                                                |
|------------------|-------------------------------------------------------------|
| `request_start`  | Start of a `/v1/chat/completions` request                   |
| `request_end`    | End of a request (streaming or non-streaming)               |
| `agent_event`    | Each AgentSoul event: token, final_token, tool_exec_*, etc. |
| `tool_call`      | MCP gateway tool execution (timing + success/failure)       |
| `llm_call`       | Direct LLM invocation with token counts                     |
| `pipeline_stage` | Pipeline stage start/end/error                              |
| `error`          | Any caught exception with component + traceback snippet     |

### Useful Query Patterns

**All events for one request** (filter by `trace_id`):
```python
import json
trace = "your-trace-id-here"
events = [json.loads(l) for l in open("logs/sessions/<uuid>.jsonl")
          if json.loads(l).get("trace_id") == trace]
```

**All errors in a session**:
```python
errors = [json.loads(l) for l in open("logs/sessions/<uuid>.jsonl")
          if json.loads(l).get("event_type") == "error"]
```

**Slowest tool calls** (sort `tool_call` by `latency_s`):
```python
import json
tools = [json.loads(l) for l in open("logs/sessions/<uuid>.jsonl")
         if json.loads(l).get("event_type") == "tool_call"]
tools.sort(key=lambda e: e.get("latency_s", 0), reverse=True)
```

**Request-level timing** (look at `request_end` events):
```python
ends = [json.loads(l) for l in open("logs/sessions/<uuid>.jsonl")
        if json.loads(l).get("event_type") == "request_end"]
# Fields: total_time_s, inference_time_s, first_token_latency_s, token_usage
```

---

## Event Schemas

### `request_start`
```json
{
  "ts": "2026-02-25T10:00:00.000Z",
  "trace_id": "uuid",
  "session_id": "uuid",
  "event_type": "request_start",
  "user_prompt_preview": "first 200 chars of user prompt",
  "model_name": "claude-sonnet-4-6",
  "route": "agent | direct | ...",
  "stream": true,
  "selected_tools": ["tool_name_1", "tool_name_2"]
}
```

### `request_end`
```json
{
  "event_type": "request_end",
  "status": "ok",
  "total_time_s": 3.45,
  "inference_time_s": 2.10,
  "first_token_latency_s": 0.82,
  "token_usage": {"prompt": 1200, "completion": 300, "total": 1500},
  "error": null
}
```

### `agent_event`
```json
{
  "event_type": "agent_event",
  "agent_event_type": "tool_exec_start",
  "payload": {"tool": "generate.from_document"}
}
```
`agent_event_type` values: `token`, `final_token`, `tool_exec_start`, `tool_exec_end`,
`phase_start`, `phase_end`, `reflection_result`, `complete`

### `tool_call`
```json
{
  "event_type": "tool_call",
  "tool_name": "generate.from_document",
  "arguments": {"technique": "sft", "file_path": "/data/doc.pdf"},
  "result_preview": "first 500 chars of result",
  "latency_s": 4.23,
  "success": true,
  "error": null
}
```

### `error`
```json
{
  "event_type": "error",
  "error_type": "TimeoutError",
  "message": "Tool call timed out after 15s",
  "component": "chat_api",
  "traceback_snippet": ["  File ...", "  ...", "TimeoutError: ..."]
}
```
`component` values: `chat_api`, `agent`, `tool`, `mcp_gateway`, `pipeline`

---

## Correlating Across Components

`trace_id` is set at the start of each HTTP request in `ChatAPIOrchestrator.handle_request()`
and propagated via Python `contextvars.ContextVar` to all `await`-ed calls in that request.

This means **all events within one request share the same `trace_id`**:
- `request_start` → `agent_event`(s) → `request_end` all have the same `trace_id`

MCP gateway tool calls (`tool_call` events) may have empty `trace_id` if the gateway
runs as a separate process (cross-process ContextVar propagation requires HTTP headers).
Correlate them by timestamp proximity or `tool_name` if `trace_id` is absent.

---

## Writing Findings

Use `AIDiagnosticWriter` to record your analysis. This is a **synchronous** writer —
no event loop needed.

```python
from shared.ai_diagnostic import AIDiagnosticWriter

w = AIDiagnosticWriter(log_root="logs")
w.write_finding(
    session_id="<uuid from filename>",
    trace_id="<trace_id from event>",
    severity="error",           # info | warning | error | critical
    category="tool_failure",    # latency | error_rate | tool_failure |
                                # token_usage | reflection_loop | other
    summary="generate.from_document fails with TimeoutError on PDFs > 50 pages",
    evidence={
        "tool_name": "generate.from_document",
        "error_type": "TimeoutError",
        "latency_s": 15.02,
        "file_path": "/data/large.pdf",
    },
    recommendation="Increase tool timeout from 15s to 60s for document generation tools",
)
```

### Finding Schema (all fields required except `recommendation`)

| Field            | Type   | Constraint                                                    |
|------------------|--------|---------------------------------------------------------------|
| `session_id`     | string | UUID from `logs/sessions/<uuid>.jsonl` filename               |
| `trace_id`       | string | `trace_id` from events, or `""` if unknown                    |
| `severity`       | enum   | `info` \| `warning` \| `error` \| `critical`                 |
| `category`       | enum   | `latency` \| `error_rate` \| `tool_failure` \| `token_usage` \| `reflection_loop` \| `other` |
| `summary`        | string | Max 200 chars                                                 |
| `evidence`       | dict   | Supporting data (latencies, counts, trace snippets)           |
| `recommendation` | string | Optional. Actionable fix.                                     |

---

## Requesting More Data

If you need information that isn't in the session logs, write a data request:

```python
w.request_data(
    description="Need the full tool arguments for all generate.from_document calls "
                "in session abc123 to check if large files are consistently slow",
    target_session_id="abc123-...",
    target_trace_id=None,             # None = all traces in the session
    filters={"event_type": "tool_call", "tool_name": "generate.from_document"},
)
```

A human operator will check `logs/ai_diagnostics/data_requests.jsonl` and fulfil pending requests.

### Data Request Schema

| Field               | Type   | Description                                       |
|---------------------|--------|---------------------------------------------------|
| `description`       | string | What you need and why                             |
| `target_session_id` | string | Session to focus on, or `null`                    |
| `target_trace_id`   | string | Specific trace to focus on, or `null`             |
| `filters`           | dict   | Field filters like `{"event_type": "tool_call"}`  |
| `status`            | string | Always `"pending"` when written                   |

---

## Rules

1. **NEVER modify `logs/agentic.log`** — it is the existing plain-text rotating log.
2. **NEVER modify session JSONL files** in `logs/sessions/` — they are append-only.
3. **NEVER log raw prompts or API keys** — `sanitize()` in `shared/diagnostics.py` redacts
   keys matching: `api_key`, `apikey`, `password`, `passwd`, `token`, `secret`,
   `authorization`, `x-api-key`.
4. Findings are identified as written by `"claude-sonnet-4-6"` automatically.
5. The `status` field in data requests is always `"pending"` — do not set it to anything else.
