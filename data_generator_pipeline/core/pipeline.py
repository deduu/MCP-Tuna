# ============================================================================
# FILE: src/finetuning/core/pipeline.py
# ============================================================================

from typing import List, Dict, Type
from dataclasses import asdict
import tqdm
from .base import BaseGenerator
from ..models.datapoints import BaseDataPoint, SFTDataPoint


class FineTuningPipeline:
    """Generic pipeline that works with any generator."""

    def __init__(
        self,
        generator: BaseGenerator,
        data_point_class: Type[BaseDataPoint],
    ):
        self.generator = generator
        self.data_point_class = data_point_class

    async def run(
        self,
        file_name: str,
        pages: List[Dict],
        max_items_per_page: int = None,
        **generator_kwargs
    ) -> List[BaseDataPoint]:
        """Process all pages and return structured data points."""
        results = []

        for page in tqdm.tqdm(pages, desc=f"Processing {file_name}"):
            try:
                items = await self.generator.generate_from_page(
                    page["markdown"],
                    **generator_kwargs
                )

                if max_items_per_page is not None:
                    items = items[:max_items_per_page]

                for item in items:
                    data_point = self.data_point_class(
                        id=len(results) + 1,
                        file_name=file_name,
                        page=page["index"] + 1,
                        text=page["markdown"],
                        **item,
                    )
                    results.append(data_point)

            except Exception as e:
                print(f"❌ Page {page['index'] + 1} failed: {e}")
                if self.generator.debug:
                    raise

        return results

    def to_dict_list(self, data_points: List[BaseDataPoint]) -> List[Dict]:
        """Convert data points to dictionaries."""
        return [asdict(dp) for dp in data_points]
