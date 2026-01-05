# core/metrics/base.py
from abc import ABC, abstractmethod
from typing import Dict
from ..data import DataPoint


class BaseMetric(ABC):
    name: str

    @abstractmethod
    def compute(self, dp: DataPoint) -> float:
        pass
