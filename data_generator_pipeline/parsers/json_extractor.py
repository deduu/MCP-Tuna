# ============================================================================
# FILE: data_generator_pipeline/parsers/json_extractor.py
#
# Thin wrapper around shared.json_extractor that satisfies the BaseParser ABC.
# ============================================================================

from ..core.base import BaseParser
from shared.json_extractor import JsonExtractor as _SharedExtractor


class JsonExtractor(BaseParser):
    """Extract JSON from LLM responses (delegates to shared.json_extractor)."""

    def __init__(self):
        self._inner = _SharedExtractor()

    def extract(self, content: str) -> list:
        return self._inner.extract(content)
