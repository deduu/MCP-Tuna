"""Orchestration MCP server wrapper.

Re-exports OrchestrationMCPServer with consistent entry point.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from orchestration.mcp.server import OrchestrationMCPServer


class OrchestrateServer:
    """Thin wrapper for consistent entry point."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._inner = OrchestrationMCPServer(config)
        self.mcp = self._inner.mcp

    def run(self, transport=None):
        self._inner.run(transport)
