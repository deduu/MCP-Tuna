# core/data.py
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DataPoint:
    instruction: str
    input: str
    output: str
    metadata: Optional[Dict] = None

    @property
    def full_instruction(self) -> str:
        return f"{self.instruction} {self.input}".strip()
