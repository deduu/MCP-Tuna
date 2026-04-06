from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuning_pipeline.services.pipeline_service import FineTuningService
from orchestration.benchmark_eval import evaluate_candidate_with_finetuner


async def _main_async(request_path: Path, response_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    finetuner = FineTuningService(
        default_base_model=str(request.get("base_model") or request.get("model_path") or "")
    )
    result = await evaluate_candidate_with_finetuner(
        finetuner=finetuner,
        model_path=str(request["model_path"]),
        adapter_path=request.get("adapter_path"),
        evaluation_packs=dict(request["evaluation_packs"]),
        eval_system_prompt=request.get("eval_system_prompt"),
        eval_quantization=request.get("eval_quantization"),
        eval_max_new_tokens=int(request["eval_max_new_tokens"]),
    )
    response_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: evaluate_benchmark_candidate.py <request_json> <response_json>")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    asyncio.run(_main_async(request_path, response_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
