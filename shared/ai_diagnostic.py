"""
AI Diagnostic Writer for MCP Tuna.

Synchronous writer that lets Claude (running as a Claude Code session)
record diagnostic findings and request additional data.

Files written:
  logs/ai_diagnostics/findings.jsonl     — analysis findings
  logs/ai_diagnostics/data_requests.jsonl — requests for more data

Usage (from a Claude Code session):
  from shared.ai_diagnostic import AIDiagnosticWriter
  w = AIDiagnosticWriter()
  w.write_finding(
      session_id="<session-uuid>",
      trace_id="<trace-uuid>",
      severity="error",
      category="latency",
      summary="P99 latency > 10s on generate.from_document",
      evidence={"max_latency_s": 12.4, "trace_id": "abc"},
      recommendation="Add timeout guard in PipelineService.generate_from_document",
  )
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional


class AIDiagnosticWriter:
    """Synchronous writer for Claude diagnostic sessions.

    Creates logs/ai_diagnostics/ on init.
    All methods are synchronous — safe to call from Claude Code sessions
    without an asyncio event loop.
    """

    _WRITTEN_BY = "claude-sonnet-4-6"

    def __init__(self, log_root: str = "logs") -> None:
        self._diag_dir = os.path.join(log_root, "ai_diagnostics")
        os.makedirs(self._diag_dir, exist_ok=True)
        self._findings_path = os.path.join(self._diag_dir, "findings.jsonl")
        self._requests_path = os.path.join(self._diag_dir, "data_requests.jsonl")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_line(self, path: str, record: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"[AIDiagnosticWriter] IOError writing to {path}: {exc}", file=sys.stderr)

    def write_finding(
        self,
        session_id: str,
        trace_id: str,
        severity: str,
        category: str,
        summary: str,
        evidence: dict,
        recommendation: Optional[str] = None,
    ) -> None:
        """Write a diagnostic finding to findings.jsonl.

        Args:
            session_id: The session UUID from logs/sessions/<uuid>.jsonl filename.
            trace_id: The trace_id from events within that session.
            severity: One of 'info', 'warning', 'error', 'critical'.
            category: One of 'latency', 'error_rate', 'tool_failure',
                      'token_usage', 'reflection_loop', 'other'.
            summary: Max 200 chars. Human-readable finding summary.
            evidence: Dict of supporting data (latencies, counts, trace snippets).
            recommendation: Optional actionable fix recommendation.
        """
        record = {
            "ts": self._ts(),
            "written_by": self._WRITTEN_BY,
            "session_id": session_id,
            "trace_id": trace_id,
            "severity": severity,
            "category": category,
            "summary": summary[:200],
            "evidence": evidence,
            "recommendation": recommendation,
        }
        self._write_line(self._findings_path, record)

    def request_data(
        self,
        description: str,
        target_session_id: Optional[str] = None,
        target_trace_id: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> None:
        """Write a data request to data_requests.jsonl.

        The backend operator (human or automated) should check this file
        and provide the requested data for further analysis.

        Args:
            description: What data is needed and why.
            target_session_id: Optional session to focus on.
            target_trace_id: Optional trace to focus on.
            filters: Optional dict of field filters (e.g. {"event_type": "tool_call"}).
        """
        record = {
            "ts": self._ts(),
            "written_by": self._WRITTEN_BY,
            "description": description,
            "target_session_id": target_session_id,
            "target_trace_id": target_trace_id,
            "filters": filters or {},
            "status": "pending",
        }
        self._write_line(self._requests_path, record)
