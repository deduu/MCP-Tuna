"""
Structured diagnostic layer for MCP Tuna.

Writes per-session JSONL files to logs/sessions/ and threads a trace_id
(UUID per request) across all async calls via ContextVars.

Usage:
  # In main.py startup:
  from shared.diagnostics import init_diagnostics, session_id_var
  session_id_var.set(str(uuid.uuid4()))
  _diag_writer = init_diagnostics(log_root="logs")

  # In request handler:
  from shared.diagnostics import trace_id_var, emit_request_start
  trace_id_var.set(str(uuid.uuid4()))
  await emit_request_start(...)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

# ──────────────────────────────────────────────────────────
# Context vars — set once per request, propagate to all awaits
# ──────────────────────────────────────────────────────────
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")

_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "api_key", "apikey", "password", "passwd",
    "token", "secret", "authorization", "x-api-key",
})

# Module-level singleton
_writer: Optional[DiagnosticWriter] = None


def sanitize(obj: Any) -> Any:
    """Recursively redact sensitive dict keys with '[REDACTED]'."""
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if k.lower() in _SENSITIVE_KEYS else sanitize(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [sanitize(item) for item in obj]
    return obj


class DiagnosticWriter:
    """Appends JSONL events to logs/sessions/<session_id>.jsonl.

    asyncio.Lock guards file I/O. Falls back to stderr on IOError (never crashes).
    Creates logs/sessions/ on init.
    """

    def __init__(self, log_root: str = "logs") -> None:
        self._log_root = log_root
        self._sessions_dir = os.path.join(log_root, "sessions")
        os.makedirs(self._sessions_dir, exist_ok=True)
        self._lock = asyncio.Lock()

    def _session_path(self) -> str:
        sid = session_id_var.get() or "unknown"
        return os.path.join(self._sessions_dir, f"{sid}.jsonl")

    async def emit(self, event: dict) -> None:
        """Add ts, trace_id, session_id; sanitize; append as JSONL."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id_var.get(),
            "session_id": session_id_var.get(),
        }
        record.update(sanitize(event))
        line = json.dumps(record) + "\n"
        path = self._session_path()
        async with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as exc:
                print(
                    f"[DiagnosticWriter] IOError writing to {path}: {exc}",
                    file=sys.stderr,
                )

    async def close(self) -> None:
        """Flush and close. No-op since we open per-write."""
        pass


# ──────────────────────────────────────────────────────────
# Module-level singleton management
# ──────────────────────────────────────────────────────────

def init_diagnostics(log_root: str = "logs") -> DiagnosticWriter:
    """Initialize module-level singleton. Call once in main.py."""
    global _writer
    _writer = DiagnosticWriter(log_root=log_root)
    return _writer


def get_writer() -> Optional[DiagnosticWriter]:
    return _writer


# ──────────────────────────────────────────────────────────
# Convenience emitters (async, no-op if writer is None)
# ──────────────────────────────────────────────────────────

async def emit_request_start(
    user_prompt_preview: str,
    model_name: str,
    route: str,
    stream: bool,
    selected_tools: list,
) -> None:
    """Emit a request_start event."""
    if _writer is None:
        return
    await _writer.emit({
        "event_type": "request_start",
        "user_prompt_preview": user_prompt_preview[:200],
        "model_name": model_name,
        "route": route,
        "stream": stream,
        "selected_tools": selected_tools,
    })


async def emit_request_end(
    status: str,
    total_time_s: float,
    inference_time_s: float,
    first_token_latency_s: float,
    token_usage: dict,
    error: Optional[str] = None,
) -> None:
    """Emit a request_end event. status: 'ok' | 'error'."""
    if _writer is None:
        return
    await _writer.emit({
        "event_type": "request_end",
        "status": status,
        "total_time_s": total_time_s,
        "inference_time_s": inference_time_s,
        "first_token_latency_s": first_token_latency_s,
        "token_usage": token_usage,
        "error": error,
    })


async def emit_llm_call(
    model_id: str,
    turn: int,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    enable_thinking: bool,
) -> None:
    """Emit an llm_call event for a single LLM invocation."""
    if _writer is None:
        return
    await _writer.emit({
        "event_type": "llm_call",
        "model_id": model_id,
        "turn": turn,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_s": latency_s,
        "enable_thinking": enable_thinking,
    })


async def emit_tool_call(
    tool_name: str,
    arguments: dict,
    result_preview: str,
    latency_s: float,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Emit a tool_call event for an MCP tool invocation."""
    if _writer is None:
        return
    await _writer.emit({
        "event_type": "tool_call",
        "tool_name": tool_name,
        "arguments": sanitize(arguments),
        "result_preview": result_preview[:500],
        "latency_s": latency_s,
        "success": success,
        "error": error,
    })


async def emit_pipeline_stage(
    stage: str,
    status: str,
    details: Optional[dict] = None,
) -> None:
    """Emit a pipeline_stage event. status: 'start' | 'end' | 'error'."""
    if _writer is None:
        return
    await _writer.emit({
        "event_type": "pipeline_stage",
        "stage": stage,
        "status": status,
        "details": details,
    })


async def emit_agent_event(
    agent_event_type: str,
    payload: Optional[dict] = None,
) -> None:
    """Emit an agent_event wrapping an AgentSoul event."""
    if _writer is None:
        return
    await _writer.emit({
        "event_type": "agent_event",
        "agent_event_type": agent_event_type,
        "payload": payload,
    })


async def emit_error(
    error_type: str,
    message: str,
    component: str,
    traceback_snippet: Optional[list[str]] = None,
) -> None:
    """Emit an error event.

    component: 'chat_api' | 'agent' | 'tool' | 'mcp_gateway' | 'pipeline'
    """
    if _writer is None:
        return
    await _writer.emit({
        "event_type": "error",
        "error_type": error_type,
        "message": message,
        "component": component,
        "traceback_snippet": traceback_snippet,
    })
