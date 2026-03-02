# Re-export unified models from shared layer + generator-specific extensions
from dataclasses import dataclass

from shared.models import (
    BaseDataPoint as _SharedBase,
    DPODataPoint,
    GRPODataPoint,
    KTODataPoint,
)


@dataclass
class BaseDataPoint(_SharedBase):
    """Generator-specific base that adds source tracking fields."""
    id: int = 0
    file_name: str = ""
    page: int = 0
    text: str = ""


@dataclass
class SFTDataPoint(BaseDataPoint):
    """Supervised Fine-Tuning format (instruction/input/output inherited from shared BaseDataPoint)."""
    pass


# Re-export so existing imports keep working
__all__ = [
    "BaseDataPoint",
    "SFTDataPoint",
    "DPODataPoint",
    "GRPODataPoint",
    "KTODataPoint",
]
