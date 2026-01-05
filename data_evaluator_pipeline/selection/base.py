# selection/base.py
from abc import ABC, abstractmethod
from typing import List
from ..core.data import DataPoint


class BaseSelector(ABC):
    @abstractmethod
    def select(self, dataset: List[DataPoint]) -> List[DataPoint]:
        pass
