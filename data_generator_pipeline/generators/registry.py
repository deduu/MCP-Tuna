# ============================================================================
# FILE: src/finetuning/generators/registry.py
# ============================================================================

from typing import Dict, Type, Tuple
from ..core.base import BaseGenerator
from ..models.datapoints import (
    BaseDataPoint,
    SFTDataPoint,
    DPODataPoint,
    GRPODataPoint,
    KTODataPoint
)
from .sft import SFTGenerator
from .dpo import DPOGenerator
from .grpo import GRPOGenerator


GENERATOR_REGISTRY: Dict[str, Tuple[Type[BaseGenerator], Type[BaseDataPoint]]] = {
    "sft": (SFTGenerator, SFTDataPoint),
    "dpo": (DPOGenerator, DPODataPoint),
    "grpo": (GRPOGenerator, GRPODataPoint),
    # Add more as needed:
    # "kto": (KTOGenerator, KTODataPoint),
}


def register_generator(
    name: str,
    generator_class: Type[BaseGenerator],
    datapoint_class: Type[BaseDataPoint]
):
    """Register a custom generator."""
    GENERATOR_REGISTRY[name] = (generator_class, datapoint_class)
