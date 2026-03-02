from typing import Dict, Type, Tuple
from ..core.base import BaseGenerator
from ..models.datapoints import (
    BaseDataPoint,
    SFTDataPoint,
    DPODataPoint,
    GRPODataPoint,
)
from .sft import SFTGenerator
from .dpo import DPOGenerator
from .grpo import GRPOGenerator

from shared.registry import generator_registry

# ---- Register generators in the shared registry ----
generator_registry.add("sft", SFTGenerator, datapoint=SFTDataPoint)
generator_registry.add("dpo", DPOGenerator, datapoint=DPODataPoint)
generator_registry.add("grpo", GRPOGenerator, datapoint=GRPODataPoint)

# Legacy dict kept for backwards compatibility with factory.py / pipeline.py
GENERATOR_REGISTRY: Dict[str, Tuple[Type[BaseGenerator], Type[BaseDataPoint]]] = {
    "sft": (SFTGenerator, SFTDataPoint),
    "dpo": (DPOGenerator, DPODataPoint),
    "grpo": (GRPOGenerator, GRPODataPoint),
}


def register_generator(
    name: str,
    generator_class: Type[BaseGenerator],
    datapoint_class: Type[BaseDataPoint],
):
    """Register a custom generator in both registries."""
    GENERATOR_REGISTRY[name] = (generator_class, datapoint_class)
    generator_registry.add(name, generator_class, datapoint=datapoint_class)
