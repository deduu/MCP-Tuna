"""Unified data models used by all MCP Tuna pipelines."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaseDataPoint:
    """Universal SFT-style data point shared across generator, evaluator, and finetuner."""
    instruction: str
    input: str = ""
    output: str = ""
    metadata: Optional[Dict] = field(default=None)

    @property
    def full_instruction(self) -> str:
        return f"{self.instruction} {self.input}".strip()


@dataclass
class DPODataPoint:
    """Direct Preference Optimization format."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Optional[Dict] = field(default=None)


@dataclass
class GRPODataPoint:
    """Group Relative Policy Optimization format."""
    prompt: str
    responses: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    metadata: Optional[Dict] = field(default=None)


@dataclass
class KTODataPoint:
    """Kahneman-Tversky Optimization format."""
    prompt: str
    completion: str
    label: bool  # True for desirable, False for undesirable
    metadata: Optional[Dict] = field(default=None)
