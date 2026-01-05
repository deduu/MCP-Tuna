import json
from pathlib import Path


class DatasetExporter:
    @staticmethod
    def to_excel(data: list, output_path: str):
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        print(f"✅ Saved {len(df)} rows to {output_path}")

    @staticmethod
    def to_jsonl(data: list, output_path: str):
        output_path = Path(output_path)

        seen = set()
        written = 0

        with output_path.open("w", encoding="utf-8") as f:
            for item in data:
                instruction = item.get("instruction", "").strip()
                output = item.get("output", "").strip()

                # basic validation
                if not instruction or not output:
                    continue

                # dedup by instruction
                if instruction in seen:
                    continue
                seen.add(instruction)

                record = {
                    "instruction": instruction,
                    "input": item.get("input", ""),
                    "output": output,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        print(f"✅ Saved {written} records to {output_path}")
