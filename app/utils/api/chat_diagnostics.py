"""
Chat request diagnostics — writes a JSON report per request to logs/chat_diag/
so we can trace exactly where tools get lost.
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DIAG_DIR = Path("logs/chat_diag")
_DIAG_DIR.mkdir(parents=True, exist_ok=True)


class ChatDiagnostics:
    def __init__(self):
        self._ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self._path = _DIAG_DIR / f"req_{self._ts}.json"
        self._data: Dict[str, Any] = {
            "timestamp": self._ts,
            "steps": [],
            "errors": [],
        }

    def step(self, label: str, detail: Any = None):
        self._data["steps"].append({
            "t": round(time.perf_counter(), 4),
            "label": label,
            "detail": _safe_serialize(detail),
        })

    def error(self, label: str, exc: Exception):
        self._data["errors"].append({
            "t": round(time.perf_counter(), 4),
            "label": label,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    def finish(self):
        try:
            self._path.write_text(json.dumps(self._data, indent=2, default=str))
        except Exception:
            pass  # never break the request


def _safe_serialize(obj: Any, depth: int = 0) -> Any:
    if depth > 3:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v, depth + 1) for v in obj]
    return str(obj)
