"""Training operations: dataset preparation and SFT/LoRA fine-tuning.

Heavy ML dependencies are imported inside methods so non-training commands can
run without torch/transformers installed.
"""

import json
import time
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import FinetuningConfig


class TrainingService:
    """Handles dataset preparation and LoRA/SFT training."""

    def __init__(self, config: FinetuningConfig = None, gpu: Any = None):
        self.config = config or FinetuningConfig()
        self.gpu = gpu

    async def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        prompt_column: str = "instruction",
        response_column: str = "response",
        rename_prompt_to: str = "prompt",
    ) -> Dict[str, Any]:
        """Prepare a HF Dataset from raw dicts."""
        try:
            from datasets import Dataset
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
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "datasets":
                return {"success": False, "error": "datasets not installed. Install: pip install datasets"}
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def load_dataset_from_file(
        self,
        file_path: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Load dataset from file, merge instruction+input → prompt."""
        try:
            from datasets import Dataset
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            if format == "json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif format == "jsonl":
                data = []
                with open(path, "r", encoding="utf-8") as f:
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
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "datasets":
                return {"success": False, "error": "datasets not installed. Install: pip install datasets"}
            return {"success": False, "error": str(e), "file_path": file_path}
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
        """Train a model with (Q)LoRA SFT using TRL's SFTTrainer."""
        start_time = time.time()
        model_name = base_model or self.config.base_model

        try:
            import torch
            from datasets import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig
            from trl import SFTTrainer
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or "a required dependency"
            return {
                "success": False,
                "error": f"Missing dependency: {missing}. Install training deps (torch, transformers, peft, trl, datasets).",
                "base_model": model_name,
                "output_dir": output_dir,
            }

        try:
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)

            if prompt_column not in dataset.column_names or response_column not in dataset.column_names:
                return {
                    "success": False,
                    "error": f"Dataset must contain '{prompt_column}' and '{response_column}' columns",
                    "columns": list(getattr(dataset, "column_names", [])),
                }

            local_files_only = bool(kwargs.pop("local_files_only", False))

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, local_files_only=local_files_only
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def to_text(example: Dict[str, Any]) -> Dict[str, str]:
                prompt = (example.get(prompt_column) or "").strip()
                response = (example.get(response_column) or "").strip()
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    text = f"User: {prompt}\nAssistant: {response}"
                return {"text": text}

            dataset = dataset.map(to_text)
            # Keep only the "text" column to avoid TRL auto-detect conflicts
            # (TRL 0.28+ looks for prompt+completion if both exist alongside text)
            cols_to_remove = [c for c in dataset.column_names if c != "text"]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)

            cuda_available = torch.cuda.is_available()
            bf16_supported = bool(
                cuda_available and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
            model_dtype = torch.bfloat16 if bf16_supported else (
                torch.float16 if cuda_available else torch.float32
            )

            quantization_config = None
            load_in_4bit = bool(kwargs.pop("load_in_4bit", True))
            if load_in_4bit:
                try:
                    # Ensure bitsandbytes runtime is actually importable.
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                except Exception:
                    quantization_config = None

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=model_dtype,
                quantization_config=quantization_config,
                local_files_only=local_files_only,
            )

            peft_config = None
            if use_lora:
                target_modules = kwargs.pop(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=target_modules,
                )

            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=int(kwargs.pop("per_device_train_batch_size", 1)),
                gradient_accumulation_steps=int(kwargs.pop("gradient_accumulation_steps", 4)),
                learning_rate=float(kwargs.pop("learning_rate", 2e-4)),
                logging_steps=int(kwargs.pop("logging_steps", 10)),
                save_steps=int(kwargs.pop("save_steps", 200)),
                save_total_limit=int(kwargs.pop("save_total_limit", 2)),
                bf16=bool(kwargs.pop("bf16", bf16_supported)),
                fp16=bool(kwargs.pop("fp16", cuda_available and not bf16_supported)),
                report_to=[],
            )

            # Keep compatibility across TRL versions by only passing supported kwargs.
            trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": dataset,
            }

            if enable_evaluation and evaluation_dataset is not None and "eval_dataset" in trainer_sig:
                trainer_kwargs["eval_dataset"] = evaluation_dataset
            if "peft_config" in trainer_sig and peft_config is not None:
                trainer_kwargs["peft_config"] = peft_config
            if "dataset_text_field" in trainer_sig:
                trainer_kwargs["dataset_text_field"] = "text"
            elif "formatting_func" in trainer_sig:
                # TRL 0.28+: dataset_text_field removed, use formatting_func instead
                trainer_kwargs["formatting_func"] = lambda examples: examples["text"]
            if "max_seq_length" in trainer_sig:
                trainer_kwargs["max_seq_length"] = int(kwargs.pop("max_seq_length", 2048))
            if "packing" in trainer_sig:
                trainer_kwargs["packing"] = bool(kwargs.pop("packing", False))
            if "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = tokenizer
            elif "processing_class" in trainer_sig:
                trainer_kwargs["processing_class"] = tokenizer

            trainer = SFTTrainer(**trainer_kwargs)

            trainer.train()
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            eval_results = None
            if save_evaluation_results:
                eval_path = Path(output_dir) / "evaluation_results.json"
                if eval_path.exists():
                    with open(eval_path, "r", encoding="utf-8") as f:
                        eval_results = json.load(f)

            del trainer, model
            if self.gpu and hasattr(self.gpu, "clear_gpu_memory"):
                self.gpu.clear_gpu_memory()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            try:
                if self.gpu and hasattr(self.gpu, "clear_gpu_memory"):
                    self.gpu.clear_gpu_memory()
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return {"success": False, "error": str(e), "output_dir": output_dir, "base_model": model_name}
