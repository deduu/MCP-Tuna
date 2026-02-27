"""Model evaluation MCP server wrapper.

Re-exports ModelEvalMCPServer (evaluate_model + judge + ft_eval).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from model_evaluator_pipeline.mcp.server import ModelEvalMCPServer


class ModelEvalServer:
    """Thin wrapper for consistent entry point."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._inner = ModelEvalMCPServer()
        self.mcp = self._inner.mcp

    def run(self, transport=None):
        self._inner.run(transport)
