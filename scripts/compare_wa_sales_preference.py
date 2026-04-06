from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.runtime_python import (
    build_reexec_command,
    resolve_repo_python,
    should_reexec_into_repo_python,
)

REEXEC_ENV_VAR = "WA_PREF_COMPARE_REEXECED"


def ensure_runtime_python() -> None:
    should_reexec, preferred_python = should_reexec_into_repo_python(
        repo_root=ROOT,
        current_executable=sys.executable,
        env_var=REEXEC_ENV_VAR,
    )
    if not should_reexec or preferred_python is None:
        return
    env = dict(os.environ)
    env[REEXEC_ENV_VAR] = "1"
    print(
        f"[{time.strftime('%H:%M:%S')}] [wa_pref_compare] "
        f"Re-running under repo venv: {preferred_python}",
        flush=True,
    )
    result = subprocess.run(
        build_reexec_command(preferred_python, Path(__file__), sys.argv[1:]),
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    raise SystemExit(result.returncode)


ensure_runtime_python()

from orchestration.workflow import PipelineOrchestrator
from finetuning_pipeline.services.pipeline_service import FineTuningService
from shared.training_defaults import (
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_PREFERENCE_GRADIENT_CHECKPOINTING,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    estimate_max_steps_for_epochs,
)


NOTEBOOK_DIR = ROOT / "notebooks" / "wa_sales"
GENERATE_SCRIPT = NOTEBOOK_DIR / "generate_preference_datasets.py"
DPO_TRAIN_SCRIPT = NOTEBOOK_DIR / "train_whatsapp_sales_agent_dpo.py"
GRPO_TRAIN_SCRIPT = NOTEBOOK_DIR / "train_whatsapp_sales_agent_grpo.py"

BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
INIT_ADAPTER_PATH = os.getenv("INIT_ADAPTER_PATH")
INIT_REFERENCE_NAME = os.getenv("INIT_REFERENCE_NAME", "sft_init")
TARGET_NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", str(DEFAULT_NUM_EPOCHS)))
PREFERENCE_LEARNING_RATE = float(
    os.getenv("LEARNING_RATE", str(DEFAULT_LEARNING_RATE))
)
PREFERENCE_BATCH_SIZE = int(
    os.getenv(
        "BATCH_SIZE",
        str(DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE),
    )
)
PREFERENCE_GRAD_ACCUM = int(
    os.getenv(
        "GRAD_ACCUM",
        str(DEFAULT_GRADIENT_ACCUMULATION_STEPS),
    )
)
PREFERENCE_LORA_R = int(os.getenv("LORA_R", str(DEFAULT_LORA_R)))
PREFERENCE_LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(DEFAULT_LORA_ALPHA)))
PREFERENCE_LORA_DROPOUT = float(
    os.getenv("LORA_DROPOUT", str(DEFAULT_LORA_DROPOUT))
)
PREFERENCE_GRADIENT_CHECKPOINTING = os.getenv(
    "GRADIENT_CHECKPOINTING",
    "1" if DEFAULT_PREFERENCE_GRADIENT_CHECKPOINTING else "0",
) == "1"
METHODS = [
    method.strip().lower()
    for method in os.getenv("METHODS", "dpo,grpo").split(",")
    if method.strip()
]
SEED = int(os.getenv("SEED", "3407"))
DPO_BETA = float(os.getenv("DPO_BETA", "0.1"))
GRPO_USE_LORA = os.getenv("GRPO_USE_LORA", "1") == "1"
GRPO_MAX_PROMPT_LENGTH = int(os.getenv("GRPO_MAX_PROMPT_LENGTH", "128"))
GRPO_MAX_COMPLETION_LENGTH = int(os.getenv("GRPO_MAX_COMPLETION_LENGTH", "32"))
GRPO_NUM_GENERATIONS = int(os.getenv("GRPO_NUM_GENERATIONS", "2"))
GRPO_BATCH_SIZE = int(os.getenv("GRPO_BATCH_SIZE", str(PREFERENCE_BATCH_SIZE)))
GRPO_GRAD_ACCUM = int(os.getenv("GRPO_GRAD_ACCUM", str(PREFERENCE_GRAD_ACCUM)))
GRPO_GENERATION_BATCH_SIZE = os.getenv("GRPO_GENERATION_BATCH_SIZE")
GRPO_STEPS_PER_GENERATION = os.getenv("GRPO_STEPS_PER_GENERATION")
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(os.getenv("RUN_DIR", str(ROOT / "output" / f"wa_sales_preference_compare_{TIMESTAMP}")))
RESULTS_PATH = RUN_DIR / "results.json"

DPO_DATASET = NOTEBOOK_DIR / "whatsapp_sales_agent_train_dpo.jsonl"
GRPO_DATASET = NOTEBOOK_DIR / "whatsapp_sales_agent_train_grpo.jsonl"
EVAL_DATASET = NOTEBOOK_DIR / "whatsapp_sales_agent_eval.jsonl"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [wa_pref_compare] {message}", flush=True)


def notebook_python() -> str:
    preferred_python = resolve_repo_python(ROOT)
    return str(preferred_python if preferred_python is not None else Path(sys.executable))


def load_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_preference_datasets() -> None:
    log("Generating WhatsApp Salestify DPO/GRPO datasets")
    env = dict(os.environ)
    env.update({"BASE_MODEL": BASE_MODEL})
    subprocess.run(
        [notebook_python(), str(GENERATE_SCRIPT)],
        cwd=str(NOTEBOOK_DIR),
        env=env,
        check=True,
    )


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_max_steps(dataset_path: Path, env_name: str) -> int:
    override = os.getenv(env_name)
    if override:
        return int(override)
    return estimate_max_steps_for_epochs(
        num_examples=count_jsonl_rows(dataset_path),
        per_device_train_batch_size=PREFERENCE_BATCH_SIZE,
        gradient_accumulation_steps=PREFERENCE_GRAD_ACCUM,
        num_epochs=TARGET_NUM_EPOCHS,
    )


def resolve_grpo_generation_controls() -> tuple[int | None, int | None]:
    generation_batch_size = (
        int(GRPO_GENERATION_BATCH_SIZE) if GRPO_GENERATION_BATCH_SIZE else None
    )
    steps_per_generation = (
        int(GRPO_STEPS_PER_GENERATION) if GRPO_STEPS_PER_GENERATION else None
    )
    if generation_batch_size is not None and steps_per_generation is not None:
        log(
            "Both GRPO_GENERATION_BATCH_SIZE and GRPO_STEPS_PER_GENERATION were set; "
            "preferring generation_batch_size and ignoring steps_per_generation."
        )
        steps_per_generation = None
    return generation_batch_size, steps_per_generation


def resolve_init_adapter_path() -> str | None:
    if not INIT_ADAPTER_PATH:
        return None
    path = Path(INIT_ADAPTER_PATH)
    if not path.exists():
        raise FileNotFoundError(f"INIT_ADAPTER_PATH not found: {path}")
    return str(path)


def train_notebook_dpo(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    init_adapter_path = resolve_init_adapter_path()
    max_steps = resolve_max_steps(DPO_DATASET, "DPO_MAX_STEPS")
    env = dict(os.environ)
    env.update(
        {
            "BASE_MODEL": BASE_MODEL,
            "TRAIN_DATA": str(DPO_DATASET),
            "OUTPUT_DIR": str(output_dir),
            "LOAD_IN_4BIT": "1",
            "MAX_STEPS": str(max_steps),
            "MAX_PROMPT_LENGTH": "384",
            "MAX_LENGTH": "512",
            "LEARNING_RATE": str(PREFERENCE_LEARNING_RATE),
            "BATCH_SIZE": str(PREFERENCE_BATCH_SIZE),
            "GRAD_ACCUM": str(PREFERENCE_GRAD_ACCUM),
            "GRADIENT_CHECKPOINTING": (
                "1" if PREFERENCE_GRADIENT_CHECKPOINTING else "0"
            ),
            "LOGGING_STEPS": "1",
            "SAVE_STEPS": "30",
            "SAVE_TOTAL_LIMIT": "2",
            "SEED": str(SEED),
            "BETA": str(DPO_BETA),
            "LORA_R": str(PREFERENCE_LORA_R),
            "LORA_ALPHA": str(PREFERENCE_LORA_ALPHA),
            "LORA_DROPOUT": str(PREFERENCE_LORA_DROPOUT),
        }
    )
    if init_adapter_path:
        env["INIT_ADAPTER_PATH"] = init_adapter_path
    start = time.time()
    subprocess.run(
        [notebook_python(), str(DPO_TRAIN_SCRIPT)],
        cwd=str(NOTEBOOK_DIR),
        env=env,
        check=True,
    )
    return {
        "method": "dpo",
        "output_dir": str(output_dir),
        "model_path": BASE_MODEL,
        "adapter_path": str(output_dir),
        "init_adapter_path": init_adapter_path,
        "max_steps": max_steps,
        "run_manifest": load_optional_json(output_dir / "run_manifest.json"),
        "training_diagnostics": load_optional_json(
            output_dir / "training_diagnostics.json"
        ),
        "dpo_preprocessing": load_optional_json(output_dir / "dpo_preprocessing.json"),
        "dpo_trainer_dataset": load_optional_json(
            output_dir / "dpo_trainer_dataset.json"
        ),
        "dpo_effective_config": load_optional_json(
            output_dir / "dpo_effective_config.json"
        ),
        "training_time_seconds": round(time.time() - start, 2),
    }


def train_notebook_grpo(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    init_adapter_path = resolve_init_adapter_path()
    max_steps = resolve_max_steps(GRPO_DATASET, "GRPO_MAX_STEPS")
    generation_batch_size, steps_per_generation = resolve_grpo_generation_controls()
    env = dict(os.environ)
    env.update(
        {
            "BASE_MODEL": BASE_MODEL,
            "TRAIN_DATA": str(GRPO_DATASET),
            "OUTPUT_DIR": str(output_dir),
            "LOAD_IN_4BIT": "1",
            "USE_LORA": "1" if GRPO_USE_LORA else "0",
            "MAX_STEPS": str(max_steps),
            "MAX_PROMPT_LENGTH": str(GRPO_MAX_PROMPT_LENGTH),
            "MAX_COMPLETION_LENGTH": str(GRPO_MAX_COMPLETION_LENGTH),
            "NUM_GENERATIONS": str(GRPO_NUM_GENERATIONS),
            "LEARNING_RATE": str(PREFERENCE_LEARNING_RATE),
            "BATCH_SIZE": str(GRPO_BATCH_SIZE),
            "GRAD_ACCUM": str(GRPO_GRAD_ACCUM),
            "LOGGING_STEPS": "1",
            "SAVE_STEPS": "20",
            "SAVE_TOTAL_LIMIT": "2",
            "SEED": str(SEED),
            "LORA_R": str(PREFERENCE_LORA_R),
            "LORA_ALPHA": str(PREFERENCE_LORA_ALPHA),
            "LORA_DROPOUT": str(PREFERENCE_LORA_DROPOUT),
        }
    )
    if init_adapter_path:
        env["INIT_ADAPTER_PATH"] = init_adapter_path
    if generation_batch_size is not None:
        env["GENERATION_BATCH_SIZE"] = str(generation_batch_size)
    if steps_per_generation is not None:
        env["STEPS_PER_GENERATION"] = str(steps_per_generation)
    start = time.time()
    subprocess.run(
        [notebook_python(), str(GRPO_TRAIN_SCRIPT)],
        cwd=str(NOTEBOOK_DIR),
        env=env,
        check=True,
    )
    return {
        "method": "grpo",
        "output_dir": str(output_dir),
        "model_path": BASE_MODEL if GRPO_USE_LORA else str(output_dir),
        "adapter_path": str(output_dir) if GRPO_USE_LORA else None,
        "init_adapter_path": init_adapter_path,
        "max_steps": max_steps,
        "reward_match_stats": load_optional_json(output_dir / "reward_match_stats.json"),
        "training_diagnostics": load_optional_json(
            output_dir / "grpo_training_diagnostics.json"
        ),
        "training_time_seconds": round(time.time() - start, 2),
    }


def train_notebook_methods() -> List[Dict[str, Any]]:
    notebook_runs: List[Dict[str, Any]] = []
    for method in METHODS:
        if method == "dpo":
            log("Training notebook DPO model")
            notebook_runs.append(train_notebook_dpo(RUN_DIR / "notebook_dpo"))
        elif method == "grpo":
            log("Training notebook GRPO model")
            notebook_runs.append(train_notebook_grpo(RUN_DIR / "notebook_grpo"))
    return notebook_runs


def build_tuna_training_methods() -> List[Dict[str, Any]]:
    init_adapter_path = resolve_init_adapter_path()
    generation_batch_size, steps_per_generation = resolve_grpo_generation_controls()
    methods: List[Dict[str, Any]] = []
    if "dpo" in METHODS:
        dpo_config = {
            "name": "mcp_dpo",
            "method": "dpo",
            "dataset_path": str(DPO_DATASET),
            "num_epochs": TARGET_NUM_EPOCHS,
            "max_steps": resolve_max_steps(DPO_DATASET, "DPO_MAX_STEPS"),
            "beta": DPO_BETA,
            "logging_steps": 1,
            "save_steps": 30,
            "save_total_limit": 2,
        }
        if init_adapter_path:
            dpo_config["adapter_path"] = init_adapter_path
        methods.append(dpo_config)
    if "grpo" in METHODS:
        grpo_config = {
            "name": "mcp_grpo",
            "method": "grpo",
            "dataset_path": str(GRPO_DATASET),
            "num_epochs": TARGET_NUM_EPOCHS,
            "use_lora": GRPO_USE_LORA,
            "max_steps": resolve_max_steps(GRPO_DATASET, "GRPO_MAX_STEPS"),
            "num_generations": GRPO_NUM_GENERATIONS,
            "max_prompt_length": GRPO_MAX_PROMPT_LENGTH,
            "max_completion_length": GRPO_MAX_COMPLETION_LENGTH,
            "per_device_train_batch_size": GRPO_BATCH_SIZE,
            "gradient_accumulation_steps": GRPO_GRAD_ACCUM,
            "logging_steps": 1,
            "save_steps": 20,
            "save_total_limit": 2,
        }
        if init_adapter_path:
            grpo_config["adapter_path"] = init_adapter_path
        if generation_batch_size is not None:
            grpo_config["generation_batch_size"] = generation_batch_size
        if steps_per_generation is not None:
            grpo_config["steps_per_generation"] = steps_per_generation
        methods.append(grpo_config)
    return methods


def build_reference_models(notebook_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    init_adapter_path = resolve_init_adapter_path()
    references: List[Dict[str, Any]] = []
    if init_adapter_path:
        references.append(
            {
                "name": INIT_REFERENCE_NAME,
                "model_path": BASE_MODEL,
                "adapter_path": init_adapter_path,
            }
        )
    for run in notebook_runs:
        references.append(
            {
                "name": f"notebook_{run['method']}",
                "model_path": run["model_path"],
                "adapter_path": run["adapter_path"],
            }
        )
    return references


async def run_tuna_benchmark(notebook_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    finetuner = FineTuningService(default_base_model=BASE_MODEL)
    orchestrator = PipelineOrchestrator(
        generator=None,
        cleaner=None,
        normalizer=None,
        evaluator=None,
        finetuner=finetuner,
        hoster=None,
    )
    default_train_dataset = str(DPO_DATASET if "dpo" in METHODS else GRPO_DATASET)
    return await orchestrator.benchmark_finetuning(
        train_dataset_path=default_train_dataset,
        output_dir=str(RUN_DIR / "benchmark"),
        base_model=BASE_MODEL,
        dev_data_path=str(EVAL_DATASET),
        training_methods=build_tuna_training_methods(),
        reference_models=build_reference_models(notebook_runs),
        seeds=[SEED],
        eval_quantization="4bit",
        eval_max_new_tokens=120,
        use_lora=True,
        lora_r=PREFERENCE_LORA_R,
        lora_alpha=PREFERENCE_LORA_ALPHA,
        lora_dropout=PREFERENCE_LORA_DROPOUT,
        load_in_4bit=True,
        learning_rate=PREFERENCE_LEARNING_RATE,
        per_device_train_batch_size=PREFERENCE_BATCH_SIZE,
        gradient_accumulation_steps=PREFERENCE_GRAD_ACCUM,
        gradient_checkpointing=PREFERENCE_GRADIENT_CHECKPOINTING,
        max_seq_length=384,
        warmup_ratio=0.0,
        weight_decay=0.01,
        save_best_model=False,
        eval_process_isolation=True,
    )


async def main() -> None:
    if not METHODS:
        raise ValueError("Set METHODS to one or more of: dpo, grpo")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    init_adapter_path = resolve_init_adapter_path()
    notebook_runs: List[Dict[str, Any]] = []
    benchmark: Dict[str, Any] | None = None
    error: str | None = None

    if init_adapter_path:
        log(f"Continuing preference tuning from SFT adapter: {init_adapter_path}")

    try:
        ensure_preference_datasets()
        notebook_runs = train_notebook_methods()
        benchmark = await run_tuna_benchmark(notebook_runs)
    except Exception as exc:
        error = str(exc)
        log(f"Comparison run failed: {error}")

    result = {
        "success": bool(benchmark and benchmark.get("success")) and error is None,
        "run_dir": str(RUN_DIR),
        "base_model": BASE_MODEL,
        "init_adapter_path": init_adapter_path,
        "methods": METHODS,
        "seed": SEED,
        "error": error,
        "run_config": {
            "num_epochs": TARGET_NUM_EPOCHS,
            "learning_rate": PREFERENCE_LEARNING_RATE,
            "per_device_train_batch_size": PREFERENCE_BATCH_SIZE,
            "gradient_accumulation_steps": PREFERENCE_GRAD_ACCUM,
            "lora_r": PREFERENCE_LORA_R,
            "lora_alpha": PREFERENCE_LORA_ALPHA,
            "lora_dropout": PREFERENCE_LORA_DROPOUT,
            "gradient_checkpointing": PREFERENCE_GRADIENT_CHECKPOINTING,
            "dpo_beta": DPO_BETA,
            "grpo_num_generations": GRPO_NUM_GENERATIONS,
            "grpo_max_prompt_length": GRPO_MAX_PROMPT_LENGTH,
            "grpo_max_completion_length": GRPO_MAX_COMPLETION_LENGTH,
            "grpo_generation_batch_size": resolve_grpo_generation_controls()[0],
            "grpo_steps_per_generation": resolve_grpo_generation_controls()[1],
            "dpo_max_steps": resolve_max_steps(DPO_DATASET, "DPO_MAX_STEPS"),
            "grpo_max_steps": resolve_max_steps(GRPO_DATASET, "GRPO_MAX_STEPS"),
        },
        "datasets": {
            "dpo": str(DPO_DATASET),
            "grpo": str(GRPO_DATASET),
            "eval": str(EVAL_DATASET),
        },
        "notebook_runs": notebook_runs,
        "benchmark": benchmark,
    }
    RESULTS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Saved comparison results -> {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
