"""
Transcendence E2E — Curriculum SFT: 3B vs 1B Comparison
=========================================================

Trains Llama-3.2-3B-Instruct with 4-bit QLoRA curriculum learning,
evaluates with the two-step pipeline, then compares side-by-side
against existing 1B flat-SFT results.

VRAM budget (RTX 3050 Ti, 4 GB):
  Model (4-bit NF4):     ~1.7 GB
  LoRA (r=8):            ~0.05 GB
  Optimizer (8-bit):     ~0.2 GB
  Activations (256 seq): ~0.3 GB  (gradient_checkpointing=True)
  Overhead:              ~0.3 GB
  Total:                 ~2.55 GB  → fits in 4 GB

Usage:
    cd transcendence
    uv run python scripts/e2e_curriculum_3b_vs_1b.py
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
OUTPUT_DIR_3B = os.path.join(AGENTY_ROOT, "output", "llama-3.2-3B-skk-curriculum")
COMPARISON_DIR = os.path.join(AGENTY_ROOT, "output", "1b_vs_3b_curriculum_comparison")

# Existing 1B results
EXISTING_1B_STEP1 = os.path.join(AGENTY_ROOT, "output", "two_step_eval_results", "step1_metric_scores.jsonl")
EXISTING_1B_STEP2 = os.path.join(AGENTY_ROOT, "output", "two_step_eval_results", "step2_ft_eval_verdicts.jsonl")
EXISTING_1B_SUMMARY = os.path.join(AGENTY_ROOT, "output", "two_step_eval_results", "step2_summary.json")
EXISTING_1B_ADAPTER = os.path.join(AGENTY_ROOT, "output", "llama-3.2-1B-skk-lora")

BASE_MODEL_1B = "meta-llama/Llama-3.2-1B-Instruct"
BASE_MODEL_3B = "meta-llama/Llama-3.2-3B-Instruct"

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


def load_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_averages(step1_results):
    """Compute average metrics from step1 results."""
    if not step1_results:
        return {}
    keys = ["rouge1", "rouge2", "rougeL", "bert_f1", "llm_judge_correctness"]
    avgs = {}
    for k in keys:
        vals = [r.get(k) for r in step1_results if isinstance(r.get(k), (int, float))]
        if vals:
            avgs[f"avg_{k}"] = round(sum(vals) / len(vals), 4)
    return avgs


def compute_failure_types(step2_results):
    """Count failure types from step2 verdicts."""
    types = {}
    for r in step2_results:
        verdict = r.get("consensus_verdict", "UNKNOWN")
        if verdict == "FAIL":
            verdicts = r.get("verdicts", [])
            ft = verdicts[0].get("failure_type", "UNKNOWN") if verdicts else "UNKNOWN"
            types[ft] = types.get(ft, 0) + 1
    return types


async def main():
    wall_start = time.time()

    print("=" * 70)
    print("  Transcendence E2E: 3B Curriculum SFT vs 1B Flat SFT Comparison")
    print("=" * 70)

    # ── Verify paths ──
    for label, path in [("Train", TRAIN_PATH), ("Val", VAL_PATH), ("Test", TEST_PATH)]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} file not found: {path}")
            sys.exit(1)

    # ── Check for existing 1B results ──
    has_1b_results = all(os.path.exists(p) for p in [EXISTING_1B_STEP1, EXISTING_1B_STEP2, EXISTING_1B_SUMMARY])
    if not has_1b_results:
        print("  WARNING: 1B evaluation results not found. Comparison will only show 3B results.")
        print(f"    Expected: {EXISTING_1B_STEP1}")

    print(f"\n  3B model:    {BASE_MODEL_3B}")
    print(f"  1B model:    {BASE_MODEL_1B}")
    print(f"  Train file:  {TRAIN_PATH}")
    print(f"  Test file:   {TEST_PATH}")
    print(f"  3B output:   {OUTPUT_DIR_3B}")
    print(f"  Comparison:  {COMPARISON_DIR}")

    # ── Initialize MCP Gateway ──
    print("\n" + "-" * 70)
    print("  [1/8] Initializing MCP Gateway...")
    print("-" * 70)

    from mcp_gateway import TranscendenceGateway
    gateway = TranscendenceGateway()
    print(f"  Registered {len(gateway.mcp._tools)} tools")

    # ── STEP 1: Preflight Check ──
    print("\n" + "-" * 70)
    print("  [2/8] Preflight Check — Can 3B 4-bit fit?")
    print("-" * 70)

    preflight = await call_tool(
        gateway, "system.preflight_check",
        model_name=BASE_MODEL_3B,
        quantization="4bit",
        batch_size=1,
        max_seq_length=256,
        gradient_checkpointing=True,
        use_lora=True,
        lora_r=8,
    )

    print(f"  Can run:          {preflight.get('can_run')}")
    print(f"  Estimated VRAM:   {preflight.get('estimated_vram_gb', '?')} GB")
    print(f"  Available VRAM:   {preflight.get('available_vram_gb', '?')} GB")
    print(f"  Headroom:         {preflight.get('headroom_gb', '?')} GB")

    breakdown = preflight.get("breakdown", {})
    for k, v in breakdown.items():
        print(f"    {k}: {v} GB")

    for rec in preflight.get("recommendations", []):
        print(f"  TIP: {rec}")
    for warn in preflight.get("warnings", []):
        print(f"  WARN: {warn}")

    if not preflight.get("can_run"):
        print("\n  ABORT: 3B model does not fit on available GPU.")
        print("  Consider reducing max_seq_length, using 1B model, or freeing GPU memory.")
        return

    # ── STEP 2: System Resources ──
    print("\n" + "-" * 70)
    print("  [3/8] System Resources")
    print("-" * 70)

    resources = await call_tool(gateway, "system.check_resources")
    gpu = resources.get("gpu", {})
    ram = resources.get("ram", {})

    if gpu.get("available"):
        print(f"  GPU:        {gpu.get('name')}")
        print(f"  VRAM:       {gpu.get('vram_total_gb', '?')} GB (free: {gpu.get('vram_free_gb', '?')} GB)")
    else:
        print("  WARNING: No CUDA GPU detected!")

    print(f"  RAM total:  {ram.get('total_gb', '?')} GB")
    print(f"  RAM free:   {ram.get('free_gb', '?')} GB")

    # ── STEP 3: Curriculum Training ──
    print("\n" + "-" * 70)
    print("  [4/8] SFT Curriculum Training (3B, 4-bit QLoRA)")
    print("-" * 70)
    print(f"  Model:                  {BASE_MODEL_3B}")
    print("  Stages:                 3 (easy -> hard)")
    print("  Epochs per stage:       1")
    print("  max_seq_length:         256")
    print("  batch_size:             1 x 8 grad_accum = effective 8")
    print("  gradient_checkpointing: True")
    print("  optim:                  paged_adamw_8bit")
    print("  learning_rate:          2e-4 (cosine, 3% warmup)")
    print()

    t0 = time.time()
    train_result = await call_tool(
        gateway, "finetune.train_curriculum",
        dataset_path=TRAIN_PATH,
        output_dir=OUTPUT_DIR_3B,
        base_model=BASE_MODEL_3B,
        num_stages=3,
        num_epochs_per_stage=1,
        difficulty_order="easy_first",
        score_column="complexity",
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
        for sr in train_result.get("stage_results", []):
            tr = sr.get("training_result", {})
            print(f"    Stage {sr.get('stage')}: success={tr.get('success', '?')}")
            if not tr.get("success"):
                print(f"      error: {tr.get('error', 'unknown')}")
        return

    # Print stage details
    for sr in train_result.get("stage_results", []):
        stage = sr.get("stage", "?")
        n = sr.get("num_examples", 0)
        score_range = sr.get("score_range", [])
        tr = sr.get("training_result", {})
        print(f"\n  Stage {stage}: {n} examples, score_range={score_range}")
        if tr.get("success"):
            print(f"    model_path: {tr.get('model_path', '?')}")
            metrics = tr.get("final_metrics", {})
            if metrics:
                print(f"    final loss: {metrics.get('train_loss', '?')}")

    final_model_path = train_result.get("final_model_path", OUTPUT_DIR_3B)
    print(f"\n  Final model: {final_model_path}")

    # ── STEP 4: Inference ──
    print("\n" + "-" * 70)
    print("  [5/8] Running inference on test set (5 samples, cleaned instructions)")
    print("-" * 70)

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
        model_path=BASE_MODEL_3B,
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

    # ── STEP 5: Metric Scoring ──
    print("\n" + "-" * 70)
    print("  [6/8] Step 1: Scoring with ROUGE + BERTScore + LLM-Judge")
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

    step1_3b_results = step1_result.get("results", []) if step1_result.get("success") else []

    if step1_3b_results:
        for i, r in enumerate(step1_3b_results):
            rouge1 = r.get("rouge1", "?")
            bf1 = r.get("bert_f1", "?")
            correct = r.get("llm_judge_correctness", "?")
            if isinstance(rouge1, float):
                print(f"    Ex {i+1}: ROUGE-1={rouge1:.3f}  BERT-F1={bf1:.3f}  correctness={correct}")
            else:
                print(f"    Ex {i+1}: {rouge1}")

    # ── STEP 6: Domain Knowledge Evaluation ──
    print("\n" + "-" * 70)
    print("  [7/8] Step 2: Domain Knowledge Evaluation (PASS/FAIL + K-Type)")
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

    step2_3b_results = step2_result.get("results", []) if step2_result.get("success") else []

    if step2_3b_results:
        for i, r in enumerate(step2_3b_results):
            v = r.get("consensus_verdict", "?")
            verdicts = r.get("verdicts", [])
            ft = verdicts[0].get("failure_type", "-") if verdicts else "-"
            reason = verdicts[0].get("reasoning", "")[:100] if verdicts else ""
            print(f"    Ex {i+1}: {v}  type={ft}")
            print(f"            {reason}...")

        summary_3b = step2_result.get("summary", {})
        print(f"\n  3B Pass rate: {summary_3b.get('pass_count', 0)}/{summary_3b.get('total_samples', 0)}"
              f" ({summary_3b.get('pass_rate', 0) * 100:.0f}%)")

    # ── STEP 7: Load 1B Results + Compare ──
    print("\n" + "-" * 70)
    print("  [8/8] Comparison: 3B Curriculum SFT vs 1B Flat SFT")
    print("-" * 70)

    # Load 1B results
    if has_1b_results:
        step1_1b_results = load_jsonl(EXISTING_1B_STEP1)
        step2_1b_results = load_jsonl(EXISTING_1B_STEP2)
        with open(EXISTING_1B_SUMMARY, encoding="utf-8") as f:
            summary_1b = json.load(f)
    else:
        step1_1b_results = []
        step2_1b_results = []
        summary_1b = {}

    # Compute averages
    avgs_1b = compute_averages(step1_1b_results)
    avgs_3b = compute_averages(step1_3b_results)

    # Domain knowledge
    pass_rate_1b = summary_1b.get("pass_rate", 0)
    pass_count_1b = summary_1b.get("pass_count", 0)
    total_1b = summary_1b.get("total_samples", len(step2_1b_results))

    summary_3b_data = step2_result.get("summary", {}) if step2_result.get("success") else {}
    pass_rate_3b = summary_3b_data.get("pass_rate", 0)
    pass_count_3b = summary_3b_data.get("pass_count", 0)
    total_3b = summary_3b_data.get("total_samples", len(step2_3b_results))

    failure_types_1b = compute_failure_types(step2_1b_results)
    failure_types_3b = compute_failure_types(step2_3b_results)

    # Print comparison table
    print("\n  ┌─────────────────────────┬───────────────┬───────────────────┐")
    print("  │ Metric                  │ 1B Flat SFT   │ 3B Curriculum SFT │")
    print("  ├─────────────────────────┼───────────────┼───────────────────┤")
    print(f"  │ Avg ROUGE-1             │ {avgs_1b.get('avg_rouge1', 'N/A'):>13} │ {avgs_3b.get('avg_rouge1', 'N/A'):>17} │")
    print(f"  │ Avg ROUGE-2             │ {avgs_1b.get('avg_rouge2', 'N/A'):>13} │ {avgs_3b.get('avg_rouge2', 'N/A'):>17} │")
    print(f"  │ Avg ROUGE-L             │ {avgs_1b.get('avg_rougeL', 'N/A'):>13} │ {avgs_3b.get('avg_rougeL', 'N/A'):>17} │")
    print(f"  │ Avg BERT-F1             │ {avgs_1b.get('avg_bert_f1', 'N/A'):>13} │ {avgs_3b.get('avg_bert_f1', 'N/A'):>17} │")
    print(f"  │ Avg LLM Correctness     │ {avgs_1b.get('avg_llm_judge_correctness', 'N/A'):>13} │ {avgs_3b.get('avg_llm_judge_correctness', 'N/A'):>17} │")
    print(f"  │ Domain Pass Rate        │ {pass_count_1b}/{total_1b} ({pass_rate_1b*100:.0f}%)     │ {pass_count_3b}/{total_3b} ({pass_rate_3b*100:.0f}%)           │")
    print("  └─────────────────────────┴───────────────┴───────────────────┘")

    if failure_types_1b or failure_types_3b:
        print("\n  Failure Type Breakdown:")
        all_types = sorted(set(list(failure_types_1b.keys()) + list(failure_types_3b.keys())))
        for ft in all_types:
            c1 = failure_types_1b.get(ft, 0)
            c3 = failure_types_3b.get(ft, 0)
            print(f"    {ft:<25} 1B: {c1}  |  3B: {c3}")

    # Determine winner
    if pass_rate_3b > pass_rate_1b:
        winner = "3b_curriculum"
        analysis = (
            f"3B curriculum SFT shows {(pass_rate_3b - pass_rate_1b)*100:.0f}% improvement "
            f"in domain knowledge pass rate over 1B flat SFT."
        )
    elif pass_rate_3b == pass_rate_1b and avgs_3b.get("avg_bert_f1", 0) > avgs_1b.get("avg_bert_f1", 0):
        winner = "3b_curriculum"
        analysis = (
            "Domain knowledge pass rates are equal, but 3B curriculum shows higher "
            "BERTScore, indicating better semantic similarity."
        )
    elif pass_rate_1b > pass_rate_3b:
        winner = "1b_flat"
        analysis = (
            f"1B flat SFT unexpectedly outperforms 3B curriculum by "
            f"{(pass_rate_1b - pass_rate_3b)*100:.0f}% in domain knowledge."
        )
    else:
        winner = "tie"
        analysis = "Both models show equivalent performance."

    print(f"\n  Winner: {winner}")
    print(f"  Analysis: {analysis}")

    # ── Export ──
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    # Export 3B step1 results
    if step1_3b_results:
        path = os.path.join(COMPARISON_DIR, "3b_step1_metric_scores.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in step1_3b_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n  Exported: {path}")

    # Export 3B step2 results
    if step2_3b_results:
        path = os.path.join(COMPARISON_DIR, "3b_step2_ft_eval_verdicts.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in step2_3b_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Exported: {path}")

    # Export comparison report
    comparison_report = {
        "models": {
            "1b_flat": {
                "name": BASE_MODEL_1B,
                "adapter_path": EXISTING_1B_ADAPTER,
                "training_type": "flat_sft",
                "step1_metrics": avgs_1b,
                "step2_domain": {
                    "pass_rate": pass_rate_1b,
                    "pass_count": pass_count_1b,
                    "total_samples": total_1b,
                    "failure_types": failure_types_1b,
                },
            },
            "3b_curriculum": {
                "name": BASE_MODEL_3B,
                "adapter_path": OUTPUT_DIR_3B,
                "training_type": "curriculum_sft",
                "step1_metrics": avgs_3b,
                "step2_domain": {
                    "pass_rate": pass_rate_3b,
                    "pass_count": pass_count_3b,
                    "total_samples": total_3b,
                    "failure_types": failure_types_3b,
                },
            },
        },
        "winner": winner,
        "analysis": analysis,
        "training_config": {
            "3b": {
                "base_model": BASE_MODEL_3B,
                "quantization": "4bit",
                "num_stages": 3,
                "epochs_per_stage": 1,
                "max_seq_length": 256,
                "batch_size": "1 x 8 grad_accum",
                "optimizer": "paged_adamw_8bit",
                "learning_rate": 2e-4,
                "scheduler": "cosine",
                "lora_r": 8,
                "gradient_checkpointing": True,
            },
        },
    }

    path = os.path.join(COMPARISON_DIR, "comparison_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)
    print(f"  Exported: {path}")

    # ── Timing ──
    wall_total = time.time() - wall_start
    print("\n" + "=" * 70)
    print("  TIMING")
    print("=" * 70)
    print(f"  Training (3B curriculum):  {fmt_time(train_time)}")
    print(f"  Inference:                 {fmt_time(inference_time)}")
    print(f"  Step 1 (metrics):          {fmt_time(step1_time)}")
    print(f"  Step 2 (verdicts):         {fmt_time(step2_time)}")
    print(f"  Total wall:                {fmt_time(wall_total)}")
    print("=" * 70)
    print("  Done!")


if __name__ == "__main__":
    asyncio.run(main())
