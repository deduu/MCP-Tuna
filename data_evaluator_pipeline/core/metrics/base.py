# core/metrics/base.py
from abc import ABC, abstractmethod
from ..data import DataPoint
from shared.registry import metric_registry


class BaseMetric(ABC):
    name: str

    @abstractmethod
    def compute(self, dp: DataPoint) -> float:
        pass

    def __init_subclass__(cls, **kwargs):
        """Auto-register concrete metric subclasses in the shared metric_registry."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name") and isinstance(cls.name, str):
            metric_registry.add(cls.name, cls)
