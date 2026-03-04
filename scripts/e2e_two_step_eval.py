"""
Transcendence E2E Test — Two-Step Evaluation Pipeline via MCP Gateway
======================================================================

Step 0: Re-run inference on the fine-tuned model with cleaned instructions
Step 1: Score outputs with ROUGE + BERTScore + LLM-Judge (evaluate_model.batch)
Step 2: Domain knowledge PASS/FAIL evaluation with ft_eval.batch

Uses the existing fine-tuned Llama-3.2-1B adapter and comparison results
(for references + instructions only). Regenerates model outputs with
instructions stripped of the agent-system "# Note:" block.

Usage:
    cd transcendence
    uv run python scripts/e2e_two_step_eval.py
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
AGENTY_ROOT = _project_root
COMPARISON_JSONL = os.path.join(
    AGENTY_ROOT, "output", "evaluation_results", "base_vs_finetuned_comparison.jsonl"
)
ADAPTER_PATH = os.path.join(AGENTY_ROOT, "output", "llama-3.2-1B-skk-lora")
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RESULTS_DIR = os.path.join(AGENTY_ROOT, "output", "two_step_eval_results")

# Pattern to strip the agent-system "# Note:" block from instructions
_NOTE_RE = re.compile(r"\n*# Note:\n.*", re.DOTALL)


def strip_agent_note(instruction: str) -> str:
    """Remove the '# Note: If the question contains technical terms...' block."""
    return _NOTE_RE.sub("", instruction).strip()


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


async def call_tool(gateway, tool_name: str, **kwargs):
    """Invoke a registered MCP tool and return the parsed JSON result."""
    func = gateway.mcp._tools[tool_name]["func"]
    raw = await func(**kwargs)
    return json.loads(raw)


async def main():
    wall_start = time.time()

    print("=" * 70)
    print("  Transcendence E2E: Two-Step Evaluation Pipeline")
    print("  (with re-generated outputs from cleaned instructions)")
    print("=" * 70)

    # ── Load existing comparison data (for instructions + references) ──
    if not os.path.exists(COMPARISON_JSONL):
        print(f"  ERROR: Comparison results not found: {COMPARISON_JSONL}")
        print("  Run e2e_finetune_evaluate.py first to generate comparison data.")
        sys.exit(1)

    comparison_data = []
    with open(COMPARISON_JSONL, encoding="utf-8") as f:
        for line in f:
            comparison_data.append(json.loads(line))

    # Clean instructions
    cleaned_instructions = [strip_agent_note(row["instruction"]) for row in comparison_data]
    references = [row["reference"] for row in comparison_data]

    print(f"\n  Loaded {len(comparison_data)} comparison results from:")
    print(f"    {COMPARISON_JSONL}")
    print("\n  Cleaned instructions (# Note: stripped):")
    for i, instr in enumerate(cleaned_instructions):
        print(f"    {i + 1}. {instr[:80]}...")

    # ── Initialize MCP Gateway ──
    print("\n" + "-" * 70)
    print("  [1/5] Initializing MCP Gateway...")
    print("-" * 70)

    from mcp_gateway import TranscendenceGateway
    gateway = TranscendenceGateway()

    tool_names = list(gateway.mcp._tools.keys())
    eval_tools = [t for t in tool_names if "evaluate_model" in t or "ft_eval" in t]
    print(f"  Registered {len(tool_names)} tools")
    print(f"  Eval tools: {eval_tools}")

    # ── STEP 0: Re-run inference with cleaned instructions ──
    print("\n" + "-" * 70)
    print("  [2/5] Step 0: Re-running inference with cleaned instructions")
    print("-" * 70)
    print(f"  Base model:    {BASE_MODEL}")
    print(f"  Adapter:       {ADAPTER_PATH}")
    print(f"  Prompts:       {len(cleaned_instructions)}")
    print()

    t0 = time.time()
    inference_result = await call_tool(
        gateway, "test.inference",
        prompts=cleaned_instructions,
        model_path=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        max_new_tokens=512,
    )
    inference_time = time.time() - t0

    print(f"\n  Inference result: success={inference_result.get('success')}")
    print(f"  Time:             {fmt_time(inference_time)}")

    if not inference_result.get("success"):
        print(f"  ERROR: {inference_result.get('error')}")
        return

    # Extract generated responses
    generated_responses = []
    for r in inference_result.get("results", []):
        generated_responses.append(r.get("response", ""))

    print(f"\n  Generated {len(generated_responses)} responses:")
    for i, resp in enumerate(generated_responses):
        print(f"    {i + 1}. {resp[:100]}...")

    # ── STEP 1: Score with ROUGE + BERTScore + LLM-Judge ──
    print("\n" + "-" * 70)
    print("  [3/5] Step 1: Scoring with ROUGE + BERTScore + LLM-Judge")
    print("-" * 70)

    step1_data = []
    for instr, ref, gen in zip(cleaned_instructions, references, generated_responses):
        step1_data.append({
            "instruction": instr,
            "output": ref,          # ground truth
            "generated": gen,       # freshly generated from cleaned instruction
        })

    print(f"  Samples:    {len(step1_data)}")
    print("  Metrics:    rouge, bertscore, llm_judge")
    print("  Note:       Using freshly generated outputs (cleaned instructions)")
    print()

    t0 = time.time()
    step1_result = await call_tool(
        gateway, "evaluate_model.batch",
        test_data=step1_data,
        metrics=["rouge", "bertscore", "llm_judge"],
        flatten=True,
    )
    step1_time = time.time() - t0

    print(f"\n  Step 1 result: success={step1_result.get('success')}")
    print(f"  Rows scored:   {step1_result.get('count', 0)}")
    print(f"  Time:          {fmt_time(step1_time)}")

    if not step1_result.get("success"):
        print(f"  ERROR: {step1_result.get('error')}")
        return

    # Print Step 1 scores
    step1_results = step1_result.get("results", [])
    print("\n  --- Step 1 Scores (flat format) ---")
    for i, r in enumerate(step1_results):
        print(f"\n  Example {i + 1}:")
        print(f"    Instruction: {r.get('instruction', '')[:80]}...")
        # ROUGE
        rouge1 = r.get("rouge1", "?")
        rouge2 = r.get("rouge2", "?")
        rougeL = r.get("rougeL", "?")
        print(f"    ROUGE:  1={rouge1:.4f}  2={rouge2:.4f}  L={rougeL:.4f}" if isinstance(rouge1, float) else f"    ROUGE:  {rouge1}")
        # BERTScore
        bf1 = r.get("bert_f1", "?")
        print(f"    BERT F1: {bf1:.4f}" if isinstance(bf1, float) else f"    BERT F1: {bf1}")
        # LLM Judge
        for criterion in ["correctness", "completeness", "factuality", "structure", "hallucination_resistance"]:
            score = r.get(f"llm_judge_{criterion}", "?")
            if score != "?":
                print(f"    {criterion}: {score}")

    # ── STEP 2: Domain Knowledge PASS/FAIL ──
    print("\n" + "-" * 70)
    print("  [4/5] Step 2: Domain Knowledge Evaluation (PASS/FAIL + K-Type)")
    print("-" * 70)

    step2_data = []
    for instr, ref, gen in zip(cleaned_instructions, references, generated_responses):
        step2_data.append({
            "instruction": instr,
            "generated": gen,
            "reference": ref,
        })

    judge_models = ["gpt-4o"]
    print(f"  Samples:       {len(step2_data)}")
    print(f"  Judge models:  {judge_models}")
    print("  Failure taxonomy: K-Gap, K-Hallucination, K-Outdated, K-Leakage, K-Instruction")
    print()

    t0 = time.time()
    step2_result = await call_tool(
        gateway, "ft_eval.batch",
        test_data=step2_data,
        judge_models=judge_models,
    )
    step2_time = time.time() - t0

    print(f"\n  Step 2 result: success={step2_result.get('success')}")
    print(f"  Rows judged:   {step2_result.get('count', 0)}")
    print(f"  Time:          {fmt_time(step2_time)}")

    if not step2_result.get("success"):
        print(f"  ERROR: {step2_result.get('error')}")
        return

    # Print per-sample verdicts
    step2_results = step2_result.get("results", [])
    print("\n  --- Step 2 Verdicts ---")
    for i, r in enumerate(step2_results):
        consensus = r.get("consensus_verdict", "?")
        verdicts = r.get("verdicts", [])
        print(f"\n  Example {i + 1}: {consensus}")
        print(f"    Instruction: {r.get('instruction', '')[:80]}...")
        for v in verdicts:
            print(f"    [{v.get('judge_model', '?')}] {v.get('verdict', '?')}"
                  f"  type={v.get('failure_type', '-')}"
                  f"  severity={v.get('severity', '-')}")
            print(f"      Reasoning: {v.get('reasoning', '')[:150]}...")

    # Print Step 2 summary
    summary = step2_result.get("summary", {})
    print("\n" + "=" * 70)
    print("  STAKEHOLDER SUMMARY")
    print("=" * 70)
    print(f"  Total samples:  {summary.get('total_samples', 0)}")
    print(f"  PASS:           {summary.get('pass_count', 0)}")
    print(f"  FAIL:           {summary.get('fail_count', 0)}")
    print(f"  Pass rate:      {summary.get('pass_rate', 0) * 100:.1f}%")

    ft_dist = summary.get("failure_type_distribution", {})
    if ft_dist:
        print("\n  Failure types:")
        for ft, count in ft_dist.items():
            print(f"    {ft}: {count}")

    sev_dist = summary.get("severity_distribution", {})
    if sev_dist:
        print("\n  Severity:")
        for sev, count in sev_dist.items():
            print(f"    {sev}: {count}")

    action_dist = summary.get("action_distribution", {})
    if action_dist:
        print("\n  Suggested actions:")
        for act, count in action_dist.items():
            print(f"    {act}: {count}")

    judge_rates = summary.get("per_judge_pass_rate", {})
    if judge_rates:
        print("\n  Per-judge pass rate:")
        for model, rate in judge_rates.items():
            print(f"    {model}: {rate * 100:.1f}%")

    # ── Export results ──
    print("\n" + "-" * 70)
    print("  [5/5] Exporting results...")
    print("-" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1 export
    step1_path = os.path.join(RESULTS_DIR, "step1_metric_scores.jsonl")
    with open(step1_path, "w", encoding="utf-8") as f:
        for row in step1_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Step 1 scores:  {step1_path}")

    # Step 2 export
    step2_path = os.path.join(RESULTS_DIR, "step2_ft_eval_verdicts.jsonl")
    with open(step2_path, "w", encoding="utf-8") as f:
        for row in step2_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Step 2 verdicts: {step2_path}")

    # Summary export
    summary_path = os.path.join(RESULTS_DIR, "step2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary:         {summary_path}")

    # ── Timing ──
    wall_total = time.time() - wall_start
    print("\n" + "=" * 70)
    print("  TIMING")
    print("=" * 70)
    print(f"  Step 0 (inference): {fmt_time(inference_time)}")
    print(f"  Step 1 (metrics):   {fmt_time(step1_time)}")
    print(f"  Step 2 (verdicts):  {fmt_time(step2_time)}")
    print(f"  Total wall:         {fmt_time(wall_total)}")
    print("=" * 70)
    print("  Done!")


if __name__ == "__main__":
    asyncio.run(main())
