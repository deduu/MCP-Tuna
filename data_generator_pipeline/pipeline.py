import tqdm
from .sft_generator import SFTGenerator


class SFTPipeline:
    def __init__(self, generator: SFTGenerator):
        self.generator = generator

    async def run(self, file_name: str, pages: list) -> list:
        results = []

        for page in tqdm.tqdm(pages):
            try:
                items = await self.generator.generate_from_page(page["markdown"])
                for item in items:
                    results.append({
                        "id": len(results) + 1,
                        "file_name": file_name,
                        "page": page["index"] + 1,
                        "text": page["markdown"],
                        **item,
                    })
            except Exception as e:
                print(f"❌ Page {page['index'] + 1} failed:", e)

        return results
