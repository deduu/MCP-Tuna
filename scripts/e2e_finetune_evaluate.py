"""
MCP Tuna E2E Test — Fine-tune Llama 3.2 3B-Instruct + Compare via MCP Gateway
================================================================================

Exercises the full MCP tool chain:
  finetune.train  →  test.compare_models (base vs fine-tuned)

Usage:
    cd mcp-tuna
    uv run python scripts/e2e_finetune_evaluate.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.normpath(
    r"C:\Users\HiDigi\OneDrive\Desktop\WebDev\Vidavox_SKK\scripts\research_notebook\(4) fine_tuning"
)
CONFIG_PATH = os.path.join(DATASET_ROOT, "skk_dataset_config.yaml")
# Use the small train set (798 rows) for practical GPU time on RTX 3050 Ti (4 GB).
# Full set (9,873 rows × 3 epochs) takes ~35h on this GPU.
TRAIN_PATH = os.path.join(DATASET_ROOT, "dataset", "ksmi_train_topic_modelled_small.jsonl")
VAL_PATH = os.path.join(DATASET_ROOT, "dataset", "ksmi_val_topic_modelled_small.jsonl")
TEST_PATH = os.path.join(DATASET_ROOT, "dataset", "ksmi_test_topic_modelled.jsonl")

AGENTY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(AGENTY_ROOT, "output", "llama-3.2-1B-skk-lora")
RESULTS_DIR = os.path.join(AGENTY_ROOT, "output", "evaluation_results")

# Use 1B model — the 3B segfaults during loading when system RAM < 4 GB free.
# 1B uses ~1 GB VRAM (4-bit) and ~2 GB system RAM, well within limits.
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ── Helpers ────────────────────────────────────────────────────────────────────

async def call_tool(gateway, tool_name: str, **kwargs):
    """Invoke a registered MCP tool and return the parsed JSON result."""
    func = gateway.mcp._tools[tool_name]["func"]
    raw = await func(**kwargs)
    return json.loads(raw)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    wall_start = time.time()

    print("=" * 70)
    print("  MCP Tuna E2E Test: Fine-tune + Evaluate via MCP Gateway")
    print("=" * 70)

    # ── Validate prerequisites ──
    for p in (CONFIG_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not os.path.exists(p):
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    prompt_template = config["prompt_template"]

    print(f"\n  Dataset:        {config['dataset_name']}")
    print(f"  Base model:     {BASE_MODEL}")
    print(f"  Train file:     {TRAIN_PATH}")
    print(f"  Val file:       {VAL_PATH}")
    print(f"  Test file:      {TEST_PATH}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Results dir:    {RESULTS_DIR}")

    # ── 1. Initialize MCP Gateway ──
    print("\n" + "-" * 70)
    print("  [1/5] Initializing MCP Gateway...")
    print("-" * 70)

    from mcp_gateway import TunaGateway
    gateway = TunaGateway()

    tool_names = list(gateway.mcp._tools.keys())
    print(f"  Registered {len(tool_names)} tools")
    print(f"  Model eval tools: {[t for t in tool_names if 'evaluate_model' in t]}")

    # ── 2. Verify dataset loads ──
    print("\n" + "-" * 70)
    print("  [2/5] Verifying dataset...")
    print("-" * 70)

    load_result = await call_tool(
        gateway, "finetune.load_dataset",
        file_path=TRAIN_PATH, format="jsonl",
    )
    print(f"  Train: success={load_result['success']}, rows={load_result.get('num_rows', '?')}")

    # ── 3. Fine-tune ──
    print("\n" + "-" * 70)
    print("  [3/5] Fine-tuning Llama 3.2 3B-Instruct with LoRA (SFT)")
    print("-" * 70)
    print(f"  Model:           {BASE_MODEL}")
    print("  Epochs:          1")
    print("  LoRA r=8, alpha=16")
    print("  Batch size:      1 × 4 grad_accum")
    print("  Max seq length:  512 (reduced for 4 GB VRAM)")
    print("  Completion-only: False (TRL formatting_func compat)")
    print("  LR scheduler:    cosine, warmup=0.03")
    print(f"  Eval file:       {VAL_PATH}")
    print()

    t0 = time.time()
    train_result = await call_tool(
        gateway, "finetune.train",
        dataset_path=TRAIN_PATH,
        output_dir=OUTPUT_DIR,
        base_model=BASE_MODEL,
        num_epochs=1,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        completion_only_loss=False,
        eval_file_path=VAL_PATH,
        early_stopping_patience=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_seq_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
    )
    train_time = time.time() - t0

    print(f"\n  Training result: success={train_result.get('success')}")
    print(f"  Training time:   {fmt_time(train_time)}")

    if not train_result.get("success"):
        print(f"  ERROR: {train_result.get('error')}")
        print("\n  Training failed. Aborting.")
        return

    # The adapter is saved in OUTPUT_DIR
    adapter_path = OUTPUT_DIR
    if train_result.get("adapter_path"):
        adapter_path = train_result["adapter_path"]
    print(f"  Adapter path:    {adapter_path}")

    # Print training metrics if available
    if "final_metrics" in train_result:
        print(f"  Final metrics:   {json.dumps(train_result['final_metrics'], indent=4)}")

    # ── 4. Prepare test data ──
    print("\n" + "-" * 70)
    print("  [4/5] Preparing test data with prompt template...")
    print("-" * 70)

    # Limit test set for practical GPU time.
    # Full 200 rows × (inference + BERTScore + LLM-Judge) ≈ 3+ hours on this GPU.
    MAX_TEST_ROWS = 5

    test_data = []
    with open(TEST_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_TEST_ROWS:
                break
            row = json.loads(line)
            # Apply the prompt template (same as the notebook)
            instruction_with_template = prompt_template.format(
                query_text=row["instruction"]
            )
            test_data.append({
                "instruction": instruction_with_template,
                "output": row["output"],  # ground truth reference
            })

    print(f"  Test examples:   {len(test_data)} (capped at {MAX_TEST_ROWS} for GPU time)")
    print(f"  Sample prompt:   {test_data[0]['instruction'][:100]}...")

    # ── 5. Compare base vs fine-tuned ──
    print("\n" + "-" * 70)
    print("  [5/5] Comparing base vs fine-tuned model on test prompts...")
    print("-" * 70)

    prompts = [row["instruction"] for row in test_data]
    references = [row["output"] for row in test_data]

    print(f"  Base model:      {BASE_MODEL}")
    print(f"  Adapter:         {adapter_path}")
    print(f"  Test prompts:    {len(prompts)}")
    print("  max_new_tokens:  256")
    print()

    t0 = time.time()
    compare_result = await call_tool(
        gateway, "test.compare_models",
        prompts=prompts,
        base_model_path=BASE_MODEL,
        finetuned_adapter_path=adapter_path,
        max_new_tokens=256,
    )
    compare_time = time.time() - t0

    print(f"\n  Result: success={compare_result.get('success')}")
    print(f"  Time:   {fmt_time(compare_time)}")

    if not compare_result.get("success"):
        print(f"  ERROR: {compare_result.get('error')}")
        return

    # ── Display and save results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)

    comparisons = compare_result.get("comparisons", [])
    results_for_export = []

    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)

    for i, comp in enumerate(comparisons):
        print(f"\n  --- Example {i+1}/{len(comparisons)} ---")
        print(f"  Question: {test_data[i]['instruction'][:100]}...")
        print(f"  Base TPS:       {comp.get('base_tps', 0):.1f} tokens/sec")
        print(f"  Fine-tuned TPS: {comp.get('finetuned_tps', 0):.1f} tokens/sec")
        print(f"  Base time:      {comp.get('base_time', 0):.1f}s")
        print(f"  Fine-tuned time:{comp.get('finetuned_time', 0):.1f}s")

        base_resp = comp.get("base_response", "")[:200]
        ft_resp = comp.get("finetuned_response", "")[:200]
        ref = references[i][:200]

        print(f"\n  [BASE]:      {base_resp}...")
        print(f"\n  [FINETUNED]: {ft_resp}...")
        print(f"\n  [REFERENCE]: {ref}...")

        results_for_export.append({
            "id": i + 1,
            "instruction": test_data[i]["instruction"],
            "reference": references[i],
            "base_response": comp.get("base_response", ""),
            "finetuned_response": comp.get("finetuned_response", ""),
            "base_time_s": comp.get("base_time", 0),
            "finetuned_time_s": comp.get("finetuned_time", 0),
            "base_tps": comp.get("base_tps", 0),
            "finetuned_tps": comp.get("finetuned_tps", 0),
        })

    # Export results
    jsonl_path = os.path.join(RESULTS_DIR, "base_vs_finetuned_comparison.jsonl")
    json_path = os.path.join(RESULTS_DIR, "base_vs_finetuned_comparison.json")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in results_for_export:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_for_export, f, indent=2, ensure_ascii=False)

    # Aggregate stats
    if results_for_export:
        import statistics
        base_times = [r["base_time_s"] for r in results_for_export]
        ft_times = [r["finetuned_time_s"] for r in results_for_export]
        base_tps_list = [r["base_tps"] for r in results_for_export]
        ft_tps_list = [r["finetuned_tps"] for r in results_for_export]

        print("\n" + "=" * 70)
        print("  AGGREGATE STATS")
        print("=" * 70)
        print(f"  {'Metric':<25} {'Base':>12} {'Fine-tuned':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'Avg gen time (s)':<25} {statistics.mean(base_times):>12.1f} {statistics.mean(ft_times):>12.1f}")
        print(f"  {'Avg tokens/sec':<25} {statistics.mean(base_tps_list):>12.1f} {statistics.mean(ft_tps_list):>12.1f}")
        print(f"  {'Avg resp length (chars)':<25} "
              f"{statistics.mean(len(r['base_response']) for r in results_for_export):>12.0f} "
              f"{statistics.mean(len(r['finetuned_response']) for r in results_for_export):>12.0f}")

    wall_total = time.time() - wall_start
    print("\n" + "=" * 70)
    print("  TIMING")
    print("=" * 70)
    print(f"  Training:    {fmt_time(train_time)}")
    print(f"  Comparison:  {fmt_time(compare_time)}")
    print(f"  Total wall:  {fmt_time(wall_total)}")

    print("\n" + "=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)
    print(f"  Adapter:     {adapter_path}")
    print(f"  JSONL:       {jsonl_path}")
    print(f"  JSON:        {json_path}")
    print("=" * 70)
    print("  Done!")


if __name__ == "__main__":
    asyncio.run(main())
