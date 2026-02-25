import argparse
import gc
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CANONICAL_VALUES = [
    "On Production",
    "Penemuan Terkonfirmasi",
    "Aliran Konklusif",
    "Sedang Berproduksi (Komersial)",
    "Memiliki Izin Berproduksi",
    "FID Disetujui",
    "Fasilitas Beroperasi",
    "1 WAP masa tenggang jika tidak berproduksi",
    "Layak Komersial",
    "Eksplorasi Selesai",
    "Eksploitasi",
    "Reserve/Cadangan",
    "Reserves update, surat edaran kepala SKK Migas",
    "Pengembangan Lapangan/Proyek Baru",
    "Lapangan Berproduksi",
    "Pengembangan Lanjut Lapangan",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("â€¢", " ").replace("•", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expected_keywords(reference: str) -> list[str]:
    ref = normalize(reference)
    keys = [k for k in CANONICAL_VALUES if normalize(k) in ref]
    # Fallback for rare lines: use first content chunk from reference.
    if not keys:
        snippet = ref[:80].strip(" .,:;")
        if snippet:
            keys = [snippet]
    return keys


def contains_all(text: str, keys: list[str]) -> bool:
    norm = normalize(text)
    return all(normalize(k) in norm for k in keys)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.inference_mode()
def infer(model, tokenizer, device: str, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def evaluate(
    base_model_path: str,
    adapter_path: str,
    eval_rows: list[dict],
    max_new_tokens: int,
) -> dict:
    prompts = [r["prompt"] for r in eval_rows]
    refs = [r["response"] for r in eval_rows]
    keys_per_row = [expected_keywords(r) for r in refs]

    base_model, tok, device = load_model(base_model_path)
    base_outputs = [infer(base_model, tok, device, p, max_new_tokens) for p in prompts]
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tuned_base, tok2, device2 = load_model(base_model_path)
    tuned_model = PeftModel.from_pretrained(tuned_base, adapter_path, local_files_only=True)
    tuned_model.eval()
    tuned_outputs = [infer(tuned_model, tok2, device2, p, max_new_tokens) for p in prompts]

    base_hits = [contains_all(o, k) for o, k in zip(base_outputs, keys_per_row)]
    tuned_hits = [contains_all(o, k) for o, k in zip(tuned_outputs, keys_per_row)]

    rows = []
    for i, (p, ref, keys, bo, to, bh, th) in enumerate(
        zip(prompts, refs, keys_per_row, base_outputs, tuned_outputs, base_hits, tuned_hits), start=1
    ):
        rows.append(
            {
                "id": i,
                "prompt": p,
                "reference": ref,
                "expected_keywords": keys,
                "base_output": bo,
                "tuned_output": to,
                "base_correct": bh,
                "tuned_correct": th,
            }
        )

    n = len(rows)
    return {
        "num_samples": n,
        "base_accuracy": sum(base_hits) / n if n else 0.0,
        "tuned_accuracy": sum(tuned_hits) / n if n else 0.0,
        "absolute_gain": (sum(tuned_hits) - sum(base_hits)) / n if n else 0.0,
        "details": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate base vs adapter on E0 holdout set.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval-jsonl", default="data/sft_e0_eval.jsonl")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", default="outputs/eval_e0_base_vs_adapter.json")
    args = parser.parse_args()

    eval_rows = load_jsonl(Path(args.eval_jsonl))[: args.samples]
    result = evaluate(args.base_model, args.adapter, eval_rows, args.max_new_tokens)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, ensure_ascii=False, indent=2))
    print(f"Wrote detailed report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
