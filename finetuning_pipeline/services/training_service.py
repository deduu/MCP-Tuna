"""Training operations: dataset preparation and model training."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import Dataset
from transfer import Trainer, SFTConfig

from .gpu_service import GPUService
from AgentY.shared.config import FinetuningConfig


class TrainingService:
    """Handles dataset preparation and LoRA/SFT training."""

    def __init__(self, config: FinetuningConfig = None, gpu: GPUService = None):
        self.config = config or FinetuningConfig()
        self.gpu = gpu or GPUService()

    async def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        prompt_column: str = "instruction",
        response_column: str = "response",
        rename_prompt_to: str = "prompt",
    ) -> Dict[str, Any]:
        """Prepare a HF Dataset from raw dicts."""
        try:
            dataset = Dataset.from_list(data)
            if prompt_column != rename_prompt_to and prompt_column in dataset.column_names:
                dataset = dataset.rename_column(prompt_column, rename_prompt_to)
            return {
                "success": True,
                "num_examples": len(dataset),
                "columns": dataset.column_names,
                "sample": dataset[0] if len(dataset) > 0 else None,
                "dataset_object": dataset,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def load_dataset_from_file(
        self,
        file_path: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Load dataset from file, merge instruction+input → prompt."""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            if format == "json":
                with open(path, "r") as f:
                    data = json.load(f)
            elif format == "jsonl":
                data = []
                with open(path, "r") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            elif format == "csv":
                import pandas as pd
                df = pd.read_csv(path)
                data = df.to_dict("records")
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}

            dataset = Dataset.from_list(data)

            if {"instruction", "input"}.issubset(dataset.column_names):
                dataset = dataset.map(
                    lambda x: {"prompt": f"{x['instruction']} {x['input']}".strip()}
                )
            if "output" in dataset.column_names:
                dataset = dataset.rename_column("output", "response")
            dataset = dataset.remove_columns(
                [c for c in ["instruction", "input"] if c in dataset.column_names]
            )

            return {
                "success": True,
                "file_path": str(path),
                "format": format,
                "num_examples": len(dataset),
                "columns": dataset.column_names,
                "sample": dataset[0] if len(dataset) > 0 else None,
                "dataset_object": dataset,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    async def train_model(
        self,
        dataset: Any,
        output_dir: str,
        base_model: Optional[str] = None,
        num_epochs: int = 3,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        prompt_column: str = "prompt",
        response_column: str = "response",
        enable_evaluation: bool = True,
        evaluation_dataset: Optional[Any] = None,
        evaluation_metrics: Optional[List[str]] = None,
        save_evaluation_results: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train a model with LoRA fine-tuning."""
        try:
            start_time = time.time()
            model_name = base_model or self.config.base_model

            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)

            if evaluation_metrics is None:
                evaluation_metrics = ["perplexity", "semantic_entropy", "token_entropy"]

            config = SFTConfig(
                model_name=model_name,
                num_epochs=num_epochs,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                prompt_column=prompt_column,
                response_column=response_column,
                output_dir=output_dir,
                enable_evaluation=enable_evaluation,
                evaluation_dataset=evaluation_dataset,
                evaluation_metrics=evaluation_metrics,
                save_evaluation_results=save_evaluation_results,
                evaluation_results_path=f"{output_dir}/evaluation_results.json",
                **kwargs,
            )

            trainer = Trainer(
                task="sft",
                config=config,
                train_dataset=dataset,
                eval_dataset=evaluation_dataset,
                evaluate_during_training=False,
            )
            trainer.train()
            trainer.save_model()

            eval_results = None
            eval_path = Path(config.evaluation_results_path)
            if eval_path.exists():
                with open(eval_path, "r") as f:
                    eval_results = json.load(f)

            del trainer
            self.gpu.clear_gpu_memory()

            training_time = time.time() - start_time

            return {
                "success": True,
                "model_path": output_dir,
                "base_model": model_name,
                "training_time_seconds": training_time,
                "config": {
                    "num_epochs": num_epochs,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                },
                "evaluation_results": eval_results,
                "num_training_examples": len(dataset),
            }
        except Exception as e:
            self.gpu.clear_gpu_memory()
            return {"success": False, "error": str(e), "output_dir": output_dir}
