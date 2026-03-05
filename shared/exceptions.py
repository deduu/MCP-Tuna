"""Typed exception hierarchy for MCP Tuna.

Services raise these; MCP gateway catches and returns structured errors.
"""
from __future__ import annotations


class TunaError(Exception):
    """Base exception for all MCP Tuna errors."""


class ResourceError(TunaError):
    """Resource constraints violated (VRAM, RAM, disk)."""


class OOMError(ResourceError):
    """CUDA out-of-memory during GPU operation."""


class DiskSpaceError(ResourceError):
    """Insufficient disk space for operation."""


class PipelineError(TunaError):
    """Base for pipeline-stage errors."""


class GenerationError(PipelineError):
    """Data generation pipeline failure."""


class EvaluationError(PipelineError):
    """Evaluation pipeline failure."""


class TrainingError(PipelineError):
    """Training pipeline failure."""


class HostingError(PipelineError):
    """Hosting/deployment pipeline failure."""


class ValidationError(TunaError):
    """Input validation failed."""


class ProviderError(TunaError):
    """LLM or external API provider failure."""


class ProviderTimeoutError(ProviderError):
    """LLM or external API call exceeded time limit."""
