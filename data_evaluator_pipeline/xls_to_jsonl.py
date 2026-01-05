import pandas as pd
import json
import os
from pathlib import Path

data_path = Path.cwd() / "data" / "sft"
xlsx_path = data_path / "data_sft.xlsx"
jsonl_path = data_path / "data_sft.jsonl"

df = pd.read_excel(xlsx_path)

# normalize columns (important!)
df["instruction"] = df["instruction"].astype(str).str.strip()
df["input"] = df.get("input", "").astype(str).str.strip()
df["output"] = df["output"].astype(str).str.strip()

# drop invalid rows
df = df[(df["instruction"] != "") & (df["output"] != "")]

# deduplicate
df = df.drop_duplicates(subset=["instruction"]).reset_index(drop=True)

written = 0
with jsonl_path.open("w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        record = {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

print(f"✅ Saved {written} valid samples to {jsonl_path}")
