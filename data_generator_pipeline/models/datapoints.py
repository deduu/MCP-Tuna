# ============================================================================
# FILE: src/finetuning/models/datapoints.py
# ============================================================================

from dataclasses import dataclass
from typing import List


@dataclass
class BaseDataPoint:
    """Base class for all fine-tuning data points."""
    id: int
    file_name: str
    page: int
    text: str


@dataclass
class SFTDataPoint(BaseDataPoint):
    """Supervised Fine-Tuning format."""
    instruction: str
    input: str
    output: str


@dataclass
class DPODataPoint(BaseDataPoint):
    """Direct Preference Optimization format."""
    prompt: str
    chosen: str
    rejected: str


@dataclass
class GRPODataPoint(BaseDataPoint):
    """Group Relative Policy Optimization format."""
    prompt: str
    responses: List[str]
    rewards: List[float]


@dataclass
class KTODataPoint(BaseDataPoint):
    """Kahneman-Tversky Optimization format."""
    prompt: str
    completion: str
    label: bool  # True for desirable, False for undesirable
