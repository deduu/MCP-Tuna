import argparse
import asyncio
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finetuning_pipeline.services.pipeline_service import FineTuningService


def _default_hf_home() -> Path:
    return Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _model_dir_from_id(model_id: str) -> Path:
    return _default_hf_home() / "hub" / ("models--" + model_id.replace("/", "--"))


def find_latest_snapshot(model_id: str) -> str | None:
    snapshots = _model_dir_from_id(model_id) / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


async def run(args: argparse.Namespace) -> int:
    svc = FineTuningService(default_base_model=args.base_model)

    load = await svc.load_dataset_from_file(args.dataset, format="jsonl")
    if not load.get("success"):
        print(load)
        return 2

    result = await svc.train_model(
        dataset=load["dataset_object"],
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
        load_in_4bit=not args.no_4bit,
    )
    print(result)
    return 0 if result.get("success") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT fine-tune Llama-3.2-3B-Instruct on a JSONL dataset.")
    parser.add_argument("--dataset", default="data/sft_test_5.jsonl", help="Path to JSONL dataset.")
    parser.add_argument("--output-dir", default="output/llama32_3b_sft_test", help="Where to write adapter/model.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune).")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading (requires more VRAM).")
    parser.add_argument("--local-files-only", action="store_true", help="Force offline loading from local HF cache.")

    default_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    default_snapshot = find_latest_snapshot(default_model_id)
    parser.add_argument(
        "--base-model",
        default=default_snapshot or default_model_id,
        help="HF model id or local snapshot directory.",
    )

    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
