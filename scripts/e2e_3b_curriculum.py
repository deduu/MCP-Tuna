"""
AgentY E2E — SFT Curriculum Training: Llama 3.2 3B on 4 GB VRAM
================================================================

Trains Llama-3.2-3B-Instruct with 4-bit QLoRA using single-pass curriculum
learning: dataset sorted by complexity (easy -> hard), trained in one SFT call.
Avoids the LoRA merge step between stages (which needs ~6 GB RAM/VRAM for 3B).

VRAM budget (RTX 3050 Ti, 4 GB):
  Model (4-bit NF4):     ~1.7 GB
  LoRA (r=8):            ~0.05 GB
  Optimizer (8-bit):     ~0.2 GB
  Activations (256 seq): ~0.3 GB  (gradient_checkpointing=True)
  Overhead:              ~0.3 GB
  Total:                 ~2.55 GB  -> fits in 4 GB

Usage:
    cd AgentY
    uv run python scripts/e2e_3b_curriculum.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.normpath(
    r"C:\Users\HiDigi\OneDrive\Desktop\WebDev\Vidavox_SKK\scripts\research_notebook\(4) fine_tuning"
)
TRAIN_PATH = os.path.join(DATASET_ROOT, "dataset", "ksmi_train_topic_modelled_small.jsonl")
VAL_PATH = os.path.join(DATASET_ROOT, "dataset", "ksmi_val_topic_modelled_small.jsonl")
TEST_PATH = os.path.join(DATASET_ROOT, "dataset", "ksmi_test_topic_modelled.jsonl")

AGENTY_ROOT = _project_root
OUTPUT_DIR = os.path.join(AGENTY_ROOT, "output", "llama-3.2-3B-skk-curriculum")
RESULTS_DIR = os.path.join(AGENTY_ROOT, "output", "3b_curriculum_eval_results")

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Strip agent-system "# Note:" block from instructions
_NOTE_RE = re.compile(r"\n*# Note:\n.*", re.DOTALL)


def strip_agent_note(instruction: str) -> str:
    return _NOTE_RE.sub("", instruction).strip()


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


async def call_tool(gateway, tool_name: str, **kwargs):
    func = gateway.mcp._tools[tool_name]["func"]
    raw = await func(**kwargs)
    return json.loads(raw)


async def main():
    wall_start = time.time()

    print("=" * 70)
    print("  AgentY E2E: 3B SFT Curriculum Training + Two-Step Evaluation")
    print("=" * 70)

    # ── Verify paths ──
    for label, path in [("Train", TRAIN_PATH), ("Val", VAL_PATH), ("Test", TEST_PATH)]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} file not found: {path}")
            sys.exit(1)

    print(f"\n  Base model:  {BASE_MODEL}")
    print(f"  Train file:  {TRAIN_PATH}")
    print(f"  Val file:    {VAL_PATH}")
    print(f"  Output dir:  {OUTPUT_DIR}")

    # ── Initialize MCP Gateway ──
    print("\n" + "-" * 70)
    print("  [1/6] Initializing MCP Gateway...")
    print("-" * 70)

    from mcp_gateway import AgentYGateway
    gateway = AgentYGateway()
    print(f"  Registered {len(gateway.mcp._tools)} tools")

    # ── Check system resources ──
    print("\n" + "-" * 70)
    print("  [2/6] Checking system resources...")
    print("-" * 70)

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU:        {gpu_name}")
        print(f"  VRAM:       {gpu_mem:.1f} GB")
    else:
        print("  WARNING: No CUDA GPU detected! Training will be very slow on CPU.")

    import psutil
    ram = psutil.virtual_memory()
    print(f"  RAM total:  {ram.total / 1024**3:.1f} GB")
    print(f"  RAM free:   {ram.available / 1024**3:.1f} GB")

    if ram.available < 3.5 * 1024**3:
        print("  WARNING: Less than 3.5 GB RAM free. 3B model loading may segfault.")
        print("  Close other applications to free RAM before proceeding.")

    # ── STEP 1: Sort dataset by complexity, then SFT in one pass ──
    print("\n" + "-" * 70)
    print("  [3/6] Single-pass Curriculum SFT (3B, 4-bit QLoRA)")
    print("-" * 70)

    # Pre-sort the dataset by complexity (easy -> hard)
    train_data = []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    train_data.sort(key=lambda x: float(x.get("complexity", 0)))
    sorted_path = os.path.join(AGENTY_ROOT, "output", "3b_train_sorted.jsonl")
    os.makedirs(os.path.dirname(sorted_path), exist_ok=True)
    with open(sorted_path, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(train_data)
    c_min = train_data[0].get("complexity", 0)
    c_max = train_data[-1].get("complexity", 0)
    print(f"  Model:                  {BASE_MODEL}")
    print(f"  Training data:          {n} examples sorted by complexity")
    print(f"  Complexity range:       {c_min:.3f} -> {c_max:.3f}")
    print(f"  Epochs:                 1")
    print(f"  max_seq_length:         256  (conservative for 4 GB VRAM)")
    print(f"  batch_size:             1 x 8 grad_accum = effective 8")
    print(f"  gradient_checkpointing: True")
    print(f"  optim:                  paged_adamw_8bit")
    print(f"  learning_rate:          2e-4 (cosine, 3% warmup)")
    print()

    t0 = time.time()
    train_result = await call_tool(
        gateway, "finetune.train",
        dataset_path=sorted_path,
        output_dir=OUTPUT_DIR,
        base_model=BASE_MODEL,
        num_epochs=1,
        use_lora=True,
        lora_r=8,
        max_seq_length=256,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )
    train_time = time.time() - t0

    print(f"\n  Training result: success={train_result.get('success')}")
    print(f"  Time:            {fmt_time(train_time)}")

    if not train_result.get("success"):
        print(f"  ERROR: {train_result.get('error')}")
        return

    print(f"  Model path:      {train_result.get('model_path', '?')}")
    metrics = train_result.get("final_metrics") or {}
    if metrics:
        print(f"  Train loss:      {metrics.get('train_loss', '?')}")

    final_model_path = train_result.get("model_path", OUTPUT_DIR)
    print(f"\n  Final model: {final_model_path}")

    # ── STEP 2: Inference with cleaned instructions ──
    print("\n" + "-" * 70)
    print("  [4/6] Running inference on test set (5 samples, cleaned instructions)")
    print("-" * 70)

    # Load test data
    test_data = []
    with open(TEST_PATH, encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_data = test_data[:5]

    cleaned_instructions = [strip_agent_note(row["instruction"]) for row in test_data]
    references = [row["output"] for row in test_data]

    print(f"  Samples: {len(cleaned_instructions)}")
    for i, instr in enumerate(cleaned_instructions):
        print(f"    {i + 1}. {instr[:80]}...")

    t0 = time.time()
    inference_result = await call_tool(
        gateway, "test.inference",
        prompts=cleaned_instructions,
        model_path=BASE_MODEL,
        adapter_path=final_model_path,
        max_new_tokens=512,
    )
    inference_time = time.time() - t0

    print(f"\n  Inference: success={inference_result.get('success')}")
    print(f"  Time:      {fmt_time(inference_time)}")

    if not inference_result.get("success"):
        print(f"  ERROR: {inference_result.get('error')}")
        return

    generated_responses = [r.get("response", "") for r in inference_result.get("results", [])]
    for i, resp in enumerate(generated_responses):
        print(f"\n  Response {i + 1}: {resp[:120]}...")

    # ── STEP 3: Metric Scoring ──
    print("\n" + "-" * 70)
    print("  [5/6] Step 1: Scoring with ROUGE + BERTScore + LLM-Judge")
    print("-" * 70)

    eval_data = []
    for instr, ref, gen in zip(cleaned_instructions, references, generated_responses):
        eval_data.append({"instruction": instr, "output": ref, "generated": gen})

    t0 = time.time()
    step1_result = await call_tool(
        gateway, "evaluate_model.batch",
        test_data=eval_data,
        metrics=["rouge", "bertscore", "llm_judge"],
        flatten=True,
    )
    step1_time = time.time() - t0

    print(f"  Step 1: success={step1_result.get('success')}, count={step1_result.get('count', 0)}")
    print(f"  Time:   {fmt_time(step1_time)}")

    if step1_result.get("success"):
        for i, r in enumerate(step1_result.get("results", [])):
            rouge1 = r.get("rouge1", "?")
            bf1 = r.get("bert_f1", "?")
            correct = r.get("llm_judge_correctness", "?")
            print(f"    Ex {i+1}: ROUGE-1={rouge1:.3f}  BERT-F1={bf1:.3f}  correctness={correct}" if isinstance(rouge1, float) else f"    Ex {i+1}: {rouge1}")

    # ── STEP 4: Domain Knowledge Evaluation ──
    print("\n" + "-" * 70)
    print("  [6/6] Step 2: Domain Knowledge Evaluation (PASS/FAIL + K-Type)")
    print("-" * 70)

    ft_data = []
    for instr, ref, gen in zip(cleaned_instructions, references, generated_responses):
        ft_data.append({"instruction": instr, "generated": gen, "reference": ref})

    t0 = time.time()
    step2_result = await call_tool(
        gateway, "ft_eval.batch",
        test_data=ft_data,
        judge_models=["gpt-4o"],
    )
    step2_time = time.time() - t0

    print(f"  Step 2: success={step2_result.get('success')}, count={step2_result.get('count', 0)}")
    print(f"  Time:   {fmt_time(step2_time)}")

    if step2_result.get("success"):
        for i, r in enumerate(step2_result.get("results", [])):
            v = r.get("consensus_verdict", "?")
            verdicts = r.get("verdicts", [])
            ft = verdicts[0].get("failure_type", "-") if verdicts else "-"
            reason = verdicts[0].get("reasoning", "")[:100] if verdicts else ""
            print(f"    Ex {i+1}: {v}  type={ft}")
            print(f"            {reason}...")

        summary = step2_result.get("summary", {})
        print(f"\n  Pass rate: {summary.get('pass_count', 0)}/{summary.get('total_samples', 0)}"
              f" ({summary.get('pass_rate', 0) * 100:.0f}%)")

    # ── Export ──
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if step1_result.get("success"):
        path = os.path.join(RESULTS_DIR, "step1_metric_scores.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in step1_result.get("results", []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n  Exported: {path}")

    if step2_result.get("success"):
        path = os.path.join(RESULTS_DIR, "step2_ft_eval_verdicts.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in step2_result.get("results", []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Exported: {path}")

        path = os.path.join(RESULTS_DIR, "step2_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(step2_result.get("summary", {}), f, indent=2, ensure_ascii=False)
        print(f"  Exported: {path}")

    # ── Timing ──
    wall_total = time.time() - wall_start
    print("\n" + "=" * 70)
    print("  TIMING")
    print("=" * 70)
    print(f"  Training (curriculum):  {fmt_time(train_time)}")
    print(f"  Inference:              {fmt_time(inference_time)}")
    print(f"  Step 1 (metrics):       {fmt_time(step1_time)}")
    print(f"  Step 2 (verdicts):      {fmt_time(step2_time)}")
    print(f"  Total wall:             {fmt_time(wall_total)}")
    print("=" * 70)
    print("  Done!")


if __name__ == "__main__":
    asyncio.run(main())
