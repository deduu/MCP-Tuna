"""
AgentY E2E Test — Compare base vs fine-tuned Llama 3.2 3B-Instruct via MCP Gateway
=====================================================================================

Uses the `test.inference` and `test.compare_models` MCP tools to compare
the base model against the LoRA-adapted model on the SKK test set.

Usage:
    cd AgentY
    uv run python scripts/e2e_compare_models.py
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
TEST_PATH = os.path.join(DATASET_ROOT, "dataset", "ksmi_test_topic_modelled.jsonl")

AGENTY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use pre-existing adapter (trained on SKK dataset in a previous run)
ADAPTER_PATH = os.path.join(AGENTY_ROOT, "output", "finetuned_llama")
RESULTS_DIR = os.path.join(AGENTY_ROOT, "output", "comparison_results")

# Use 1B — 3B segfaults when system RAM < 4 GB free
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TEST_ROWS = 5
MAX_NEW_TOKENS = 256


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
    print("  AgentY E2E: Base vs Fine-tuned Model Comparison")
    print("=" * 70)

    # ── Load config + test data ──
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    prompt_template = config["prompt_template"]

    prompts = []
    references = []
    raw_instructions = []
    with open(TEST_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_TEST_ROWS:
                break
            row = json.loads(line)
            templated = prompt_template.format(query_text=row["instruction"])
            prompts.append(templated)
            references.append(row["output"])
            raw_instructions.append(row["instruction"])

    print(f"\n  Base model:      {BASE_MODEL}")
    print(f"  Adapter:         {ADAPTER_PATH}")
    print(f"  Test prompts:    {len(prompts)}")
    print(f"  Max new tokens:  {MAX_NEW_TOKENS}")

    # ── Initialize gateway ──
    print("\n  Initializing MCP Gateway...")
    from mcp_gateway import AgentYGateway
    gateway = AgentYGateway()
    print(f"  Tools: {len(gateway.mcp._tools)} registered")

    # ── Compare models ──
    print("\n" + "-" * 70)
    print("  Running test.compare_models (base vs fine-tuned)...")
    print("-" * 70)

    t0 = time.time()
    compare_result = await call_tool(
        gateway, "test.compare_models",
        prompts=prompts,
        base_model_path=BASE_MODEL,
        finetuned_adapter_path=ADAPTER_PATH,
        max_new_tokens=MAX_NEW_TOKENS,
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
        print(f"  Question: {raw_instructions[i][:100]}...")
        print(f"  Base TPS:       {comp.get('base_tps', 0):.1f} tokens/sec")
        print(f"  Fine-tuned TPS: {comp.get('finetuned_tps', 0):.1f} tokens/sec")
        print(f"  Base time:      {comp.get('base_time', 0):.1f}s")
        print(f"  Fine-tuned time:{comp.get('finetuned_time', 0):.1f}s")

        # Truncate for display
        base_resp = comp.get("base_response", "")[:200]
        ft_resp = comp.get("finetuned_response", "")[:200]
        ref = references[i][:200]

        print(f"\n  [BASE]:      {base_resp}...")
        print(f"\n  [FINETUNED]: {ft_resp}...")
        print(f"\n  [REFERENCE]: {ref}...")

        results_for_export.append({
            "id": i + 1,
            "instruction": raw_instructions[i],
            "reference": references[i],
            "base_response": comp.get("base_response", ""),
            "finetuned_response": comp.get("finetuned_response", ""),
            "base_time_s": comp.get("base_time", 0),
            "finetuned_time_s": comp.get("finetuned_time", 0),
            "base_tps": comp.get("base_tps", 0),
            "finetuned_tps": comp.get("finetuned_tps", 0),
        })

    # ── Export ──
    jsonl_path = os.path.join(RESULTS_DIR, "base_vs_finetuned_comparison.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in results_for_export:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Also save as JSON for easy reading
    json_path = os.path.join(RESULTS_DIR, "base_vs_finetuned_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_for_export, f, indent=2, ensure_ascii=False)

    # ── Aggregate stats ──
    base_times = [r["base_time_s"] for r in results_for_export]
    ft_times = [r["finetuned_time_s"] for r in results_for_export]
    base_tps_list = [r["base_tps"] for r in results_for_export]
    ft_tps_list = [r["finetuned_tps"] for r in results_for_export]

    print("\n" + "=" * 70)
    print("  AGGREGATE STATS")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Base':>12} {'Fine-tuned':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    if base_times:
        import statistics
        print(f"  {'Avg gen time (s)':<25} {statistics.mean(base_times):>12.1f} {statistics.mean(ft_times):>12.1f}")
        print(f"  {'Avg tokens/sec':<25} {statistics.mean(base_tps_list):>12.1f} {statistics.mean(ft_tps_list):>12.1f}")
        print(f"  {'Avg resp length (chars)':<25} "
              f"{statistics.mean(len(r['base_response']) for r in results_for_export):>12.0f} "
              f"{statistics.mean(len(r['finetuned_response']) for r in results_for_export):>12.0f}")

    wall_total = time.time() - wall_start
    print(f"\n  Total wall time: {fmt_time(wall_total)}")
    print(f"\n  Results saved to:")
    print(f"    {jsonl_path}")
    print(f"    {json_path}")
    print("=" * 70)
    print("  Done!")


if __name__ == "__main__":
    asyncio.run(main())
