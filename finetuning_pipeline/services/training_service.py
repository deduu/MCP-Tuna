"""Training operations: dataset preparation and SFT/DPO/GRPO/KTO fine-tuning.

Heavy ML dependencies are imported inside methods so non-training commands can
run without torch/transformers installed.
"""
from __future__ import annotations

import json
import time
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from shared.config import FinetuningConfig


class TrainingService:
    """Handles dataset preparation and LoRA/SFT/DPO/GRPO/KTO training."""

    def __init__(self, config: FinetuningConfig = None, gpu: Any = None):
        self.config = config or FinetuningConfig()
        self.gpu = gpu

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _resolve_checkpoint(
        self, output_dir: str, resume_from_checkpoint: Any
    ) -> Optional[str]:
        """Resolve 'latest' → most recent checkpoint dir, or return path as-is."""
        if not resume_from_checkpoint:
            return None
        if resume_from_checkpoint is True or str(resume_from_checkpoint) == "latest":
            checkpoints = sorted(
                [p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()],
                key=lambda p: int(p.name.split("-")[-1])
                if p.name.split("-")[-1].isdigit()
                else 0,
            )
            return str(checkpoints[-1]) if checkpoints else None
        return str(resume_from_checkpoint)

    def _detect_precision(self) -> tuple[bool, bool]:
        """Returns (cuda_available, bf16_supported)."""
        try:
            import torch
            cuda = torch.cuda.is_available()
            bf16 = cuda and bool(
                getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
            return cuda, bf16
        except Exception:
            return False, False

    def _pop_training_kwargs(
        self, kwargs: dict, cuda_available: bool, bf16_supported: bool
    ) -> dict:
        """Extract standard TrainingArguments kwargs from **kwargs."""
        report_to = kwargs.pop("report_to", [])
        if isinstance(report_to, str):
            report_to = [report_to] if report_to else []
        return {
            "per_device_train_batch_size": int(
                kwargs.pop("per_device_train_batch_size", 1)
            ),
            "per_device_eval_batch_size": int(
                kwargs.pop("per_device_eval_batch_size", 1)
            ),
            "gradient_accumulation_steps": int(
                kwargs.pop("gradient_accumulation_steps", 4)
            ),
            "learning_rate": float(kwargs.pop("learning_rate", 2e-4)),
            "weight_decay": float(kwargs.pop("weight_decay", 0.0)),
            "max_grad_norm": float(kwargs.pop("max_grad_norm", 1.0)),
            "lr_scheduler_type": str(kwargs.pop("lr_scheduler_type", "linear")),
            "warmup_ratio": float(kwargs.pop("warmup_ratio", 0.0)),
            "logging_steps": int(kwargs.pop("logging_steps", 10)),
            "save_steps": int(kwargs.pop("save_steps", 200)),
            "save_total_limit": int(kwargs.pop("save_total_limit", 2)),
            "bf16": bool(kwargs.pop("bf16", bf16_supported)),
            "fp16": bool(kwargs.pop("fp16", cuda_available and not bf16_supported)),
            "report_to": report_to,
            "seed": int(kwargs.pop("seed", 42)),
        }

    def _build_config(
        self,
        ConfigClass: Any,
        output_dir: str,
        num_epochs: int,
        has_eval: bool,
        save_best_model: bool,
        training_kwargs: dict,
        extra_kwargs: Optional[dict] = None,
    ) -> Any:
        """Instantiate a TrainingArguments subclass with checkpointing support."""
        load_best = save_best_model and has_eval
        config_kwargs: dict = {
            **training_kwargs,
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "save_strategy": "steps",
            "eval_strategy": "steps" if has_eval else "no",
            "load_best_model_at_end": load_best,
        }
        if load_best:
            config_kwargs["metric_for_best_model"] = "eval_loss"
            config_kwargs["greater_is_better"] = False
        if extra_kwargs:
            config_kwargs.update(extra_kwargs)

        # eval_strategy was renamed from evaluation_strategy in newer transformers
        sig = inspect.signature(ConfigClass.__init__).parameters
        if "eval_strategy" not in sig and "evaluation_strategy" in sig:
            config_kwargs["evaluation_strategy"] = config_kwargs.pop("eval_strategy")
        elif "eval_strategy" not in sig and "evaluation_strategy" not in sig:
            config_kwargs.pop("eval_strategy", None)

        return ConfigClass(**config_kwargs)

    def _build_lora_config(
        self, kwargs: dict, lora_r: int, lora_alpha: int, lora_dropout: float
    ) -> Any:
        from peft import LoraConfig

        target_modules = kwargs.pop(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        return LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    def _load_model_and_tokenizer(
        self, model_name: str, kwargs: dict
    ) -> tuple[Any, Any]:
        """Load tokenizer + model with optional 4-bit quantization."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_files_only = bool(kwargs.pop("local_files_only", False))
        load_in_4bit = bool(kwargs.pop("load_in_4bit", True))
        cuda_available, bf16_supported = self._detect_precision()
        model_dtype = (
            torch.bfloat16
            if bf16_supported
            else (torch.float16 if cuda_available else torch.float32)
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=local_files_only
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            try:
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

        # Prepare quantized model for stable gradient computation
        if quantization_config is not None:
            try:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model)
            except ImportError:
                pass

        return model, tokenizer

    def _cleanup(self, trainer: Any, model: Any) -> None:
        """Free GPU memory after training."""
        try:
            del trainer
            del model
        except Exception:
            pass
        if self.gpu and hasattr(self.gpu, "clear_gpu_memory"):
            self.gpu.clear_gpu_memory()
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _save_on_interrupt(
        self, trainer: Any, tokenizer: Any, output_dir: str
    ) -> None:
        """Best-effort checkpoint save when training is interrupted."""
        try:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Dataset operations
    # ----------------------------------------------------------------

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
                return {
                    "success": False,
                    "error": "datasets not installed. Install: pip install datasets",
                }
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
                return {
                    "success": False,
                    "error": "datasets not installed. Install: pip install datasets",
                }
            return {"success": False, "error": str(e), "file_path": file_path}
        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    # ----------------------------------------------------------------
    # SFT training
    # ----------------------------------------------------------------

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
        eval_file_path: Optional[str] = None,
        evaluation_metrics: Optional[List[str]] = None,
        save_evaluation_results: bool = True,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        save_best_model: bool = True,
        completion_only_loss: bool = True,
        early_stopping_patience: Optional[int] = None,
        push_to_hub: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train a model with (Q)LoRA SFT using TRL's SFTTrainer.

        Pass resume_from_checkpoint='latest' to auto-resume from the most recent
        checkpoint in output_dir, or provide an explicit path.

        Args:
            completion_only_loss: Train only on assistant tokens (default True).
            early_stopping_patience: Stop after N eval steps without improvement.
            eval_file_path: Path to a JSONL/JSON eval dataset file.
            push_to_hub: HuggingFace repo ID to push the trained model to.
        """
        start_time = time.time()
        model_name = base_model or self.config.base_model

        try:
            from datasets import Dataset
            from trl import SFTConfig, SFTTrainer
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or "a required dependency"
            return {
                "success": False,
                "error": (
                    f"Missing dependency: {missing}. "
                    "Install training deps (torch, transformers, peft, trl, datasets)."
                ),
                "base_model": model_name,
                "output_dir": output_dir,
            }

        try:
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)

            if (
                prompt_column not in dataset.column_names
                or response_column not in dataset.column_names
            ):
                return {
                    "success": False,
                    "error": (
                        f"Dataset must contain '{prompt_column}' and"
                        f" '{response_column}' columns"
                    ),
                    "columns": list(getattr(dataset, "column_names", [])),
                }

            # Load eval dataset from file if provided (and no in-memory eval dataset)
            if evaluation_dataset is None and eval_file_path:
                eval_result = await self.load_dataset_from_file(
                    eval_file_path,
                    format="jsonl" if eval_file_path.endswith(".jsonl") else "json",
                )
                if eval_result.get("success"):
                    evaluation_dataset = eval_result["dataset_object"]

            model, tokenizer = self._load_model_and_tokenizer(model_name, kwargs)
            cuda_available, bf16_supported = self._detect_precision()

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
            cols_to_remove = [c for c in dataset.column_names if c != "text"]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)

            # Format eval dataset the same way
            if evaluation_dataset is not None:
                if isinstance(evaluation_dataset, list):
                    evaluation_dataset = Dataset.from_list(evaluation_dataset)
                if "text" not in evaluation_dataset.column_names:
                    evaluation_dataset = evaluation_dataset.map(to_text)
                    eval_cols = [c for c in evaluation_dataset.column_names if c != "text"]
                    if eval_cols:
                        evaluation_dataset = evaluation_dataset.remove_columns(eval_cols)

            peft_config = None
            if use_lora:
                peft_config = self._build_lora_config(kwargs, lora_r, lora_alpha, lora_dropout)

            has_eval = enable_evaluation and evaluation_dataset is not None
            training_kwargs = self._pop_training_kwargs(kwargs, cuda_available, bf16_supported)

            # Build SFTConfig-specific extra kwargs
            sft_extra: dict = {}
            sft_config_sig = inspect.signature(SFTConfig.__init__).parameters
            if "completion_only_loss" in sft_config_sig:
                sft_extra["completion_only_loss"] = completion_only_loss

            training_args = self._build_config(
                SFTConfig,
                output_dir=output_dir,
                num_epochs=num_epochs,
                has_eval=has_eval,
                save_best_model=save_best_model,
                training_kwargs=training_kwargs,
                extra_kwargs=sft_extra if sft_extra else None,
            )

            checkpoint = self._resolve_checkpoint(output_dir, resume_from_checkpoint)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Build trainer kwargs respecting TRL version differences
            trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
            trainer_kwargs: dict = {
                "model": model,
                "args": training_args,
                "train_dataset": dataset,
            }
            if has_eval and "eval_dataset" in trainer_sig:
                trainer_kwargs["eval_dataset"] = evaluation_dataset
            if "peft_config" in trainer_sig and peft_config is not None:
                trainer_kwargs["peft_config"] = peft_config
            if "dataset_text_field" in trainer_sig:
                trainer_kwargs["dataset_text_field"] = "text"
            elif "formatting_func" in trainer_sig:
                trainer_kwargs["formatting_func"] = lambda examples: examples["text"]
            if "max_seq_length" in trainer_sig:
                trainer_kwargs["max_seq_length"] = int(kwargs.pop("max_seq_length", 2048))
            if "packing" in trainer_sig:
                trainer_kwargs["packing"] = bool(kwargs.pop("packing", False))
            if "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = tokenizer
            elif "processing_class" in trainer_sig:
                trainer_kwargs["processing_class"] = tokenizer

            # Build callbacks
            callbacks = []
            if early_stopping_patience is not None and has_eval:
                from transformers import EarlyStoppingCallback

                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
                )
            if callbacks and "callbacks" in trainer_sig:
                trainer_kwargs["callbacks"] = callbacks

            trainer = SFTTrainer(**trainer_kwargs)

            interrupted = False
            try:
                trainer.train(resume_from_checkpoint=checkpoint)
            except KeyboardInterrupt:
                interrupted = True
                self._save_on_interrupt(trainer, tokenizer, output_dir)

            if not interrupted:
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)

            # Push to HuggingFace Hub if requested
            hub_url = None
            if push_to_hub and not interrupted:
                try:
                    model.push_to_hub(push_to_hub)
                    tokenizer.push_to_hub(push_to_hub)
                    hub_url = f"https://huggingface.co/{push_to_hub}"
                except Exception:
                    hub_url = None

            eval_results = None
            if save_evaluation_results:
                eval_path = Path(output_dir) / "evaluation_results.json"
                if eval_path.exists():
                    with open(eval_path, "r", encoding="utf-8") as f:
                        eval_results = json.load(f)

            self._cleanup(trainer, model)

            result: Dict[str, Any] = {
                "success": True,
                "interrupted": interrupted,
                "model_path": output_dir,
                "base_model": model_name,
                "training_time_seconds": time.time() - start_time,
                "config": {
                    "trainer": "sft",
                    "num_epochs": num_epochs,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "completion_only_loss": completion_only_loss,
                    "resumed_from": checkpoint,
                },
                "evaluation_results": eval_results,
                "num_training_examples": len(dataset),
            }
            if hub_url:
                result["hub_url"] = hub_url
            return result
        except Exception as e:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir,
                "base_model": model_name,
            }

    # ----------------------------------------------------------------
    # DPO training
    # ----------------------------------------------------------------

    async def train_dpo_model(
        self,
        dataset: Any,
        output_dir: str,
        base_model: Optional[str] = None,
        num_epochs: int = 3,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        beta: float = 0.1,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        save_best_model: bool = True,
        evaluation_dataset: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train with DPO (Direct Preference Optimization).

        Dataset must have 'prompt', 'chosen', 'rejected' columns.
        beta controls the KL-divergence penalty (lower = more deviation allowed).
        """
        start_time = time.time()
        model_name = base_model or self.config.base_model

        try:
            from datasets import Dataset
            from trl import DPOConfig, DPOTrainer
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or "a required dependency"
            return {
                "success": False,
                "error": f"Missing dependency: {missing}.",
                "base_model": model_name,
                "output_dir": output_dir,
            }

        try:
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)

            required = {"prompt", "chosen", "rejected"}
            missing_cols = required - set(dataset.column_names)
            if missing_cols:
                return {
                    "success": False,
                    "error": f"DPO dataset missing columns: {missing_cols}",
                    "columns": list(dataset.column_names),
                }

            model, tokenizer = self._load_model_and_tokenizer(model_name, kwargs)
            cuda_available, bf16_supported = self._detect_precision()

            peft_config = None
            if use_lora:
                peft_config = self._build_lora_config(kwargs, lora_r, lora_alpha, lora_dropout)

            has_eval = evaluation_dataset is not None
            training_kwargs = self._pop_training_kwargs(kwargs, cuda_available, bf16_supported)
            args = self._build_config(
                DPOConfig,
                output_dir=output_dir,
                num_epochs=num_epochs,
                has_eval=has_eval,
                save_best_model=save_best_model,
                training_kwargs=training_kwargs,
                extra_kwargs={"beta": beta},
            )

            checkpoint = self._resolve_checkpoint(output_dir, resume_from_checkpoint)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
            trainer_kwargs: dict = {
                "model": model,
                "args": args,
                "train_dataset": dataset,
            }
            if has_eval and "eval_dataset" in trainer_sig:
                trainer_kwargs["eval_dataset"] = evaluation_dataset
            if "peft_config" in trainer_sig and peft_config is not None:
                trainer_kwargs["peft_config"] = peft_config
            if "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = tokenizer
            elif "processing_class" in trainer_sig:
                trainer_kwargs["processing_class"] = tokenizer

            trainer = DPOTrainer(**trainer_kwargs)

            interrupted = False
            try:
                trainer.train(resume_from_checkpoint=checkpoint)
            except KeyboardInterrupt:
                interrupted = True
                self._save_on_interrupt(trainer, tokenizer, output_dir)

            if not interrupted:
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)

            self._cleanup(trainer, model)

            return {
                "success": True,
                "interrupted": interrupted,
                "model_path": output_dir,
                "base_model": model_name,
                "training_time_seconds": time.time() - start_time,
                "config": {
                    "trainer": "dpo",
                    "num_epochs": num_epochs,
                    "beta": beta,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "resumed_from": checkpoint,
                },
                "num_training_examples": len(dataset),
            }
        except Exception as e:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir,
                "base_model": model_name,
            }

    # ----------------------------------------------------------------
    # GRPO training
    # ----------------------------------------------------------------

    async def train_grpo_model(
        self,
        dataset: Any,
        output_dir: str,
        base_model: Optional[str] = None,
        num_epochs: int = 3,
        num_generations: int = 4,
        max_prompt_length: int = 512,
        max_completion_length: int = 256,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train with GRPO (Group Relative Policy Optimization).

        Dataset must have 'prompt', 'responses' (list), 'rewards' (list of float)
        columns. Pre-computed rewards are used as the reward signal: the trainer
        generates completions and scores them against the stored reward table.
        """
        start_time = time.time()
        model_name = base_model or self.config.base_model

        try:
            from datasets import Dataset
            from trl import GRPOConfig, GRPOTrainer
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or "a required dependency"
            return {
                "success": False,
                "error": f"Missing dependency: {missing}.",
                "base_model": model_name,
                "output_dir": output_dir,
            }

        try:
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)

            required = {"prompt", "responses", "rewards"}
            missing_cols = required - set(dataset.column_names)
            if missing_cols:
                return {
                    "success": False,
                    "error": f"GRPO dataset missing columns: {missing_cols}",
                    "columns": list(dataset.column_names),
                }

            # Build lookup from pre-computed rewards: "{prompt}|||{response}" → reward
            reward_table: Dict[str, float] = {}
            for example in dataset:
                prompt_text = example.get("prompt", "")
                responses: list = example.get("responses") or []
                rewards: list = example.get("rewards") or []
                for resp, rew in zip(responses, rewards):
                    reward_table[f"{prompt_text}|||{resp}"] = float(rew)

            def reward_fn(
                prompts: List[str], completions: List[str], **_kw: Any
            ) -> List[float]:
                return [
                    reward_table.get(f"{p}|||{c}", 0.0)
                    for p, c in zip(prompts, completions)
                ]

            # GRPOTrainer only needs the 'prompt' column; it generates completions
            grpo_dataset = dataset.remove_columns(
                [c for c in dataset.column_names if c != "prompt"]
            )

            model, tokenizer = self._load_model_and_tokenizer(model_name, kwargs)
            cuda_available, bf16_supported = self._detect_precision()
            training_kwargs = self._pop_training_kwargs(kwargs, cuda_available, bf16_supported)

            # trl 0.28 GRPOConfig does not have max_prompt_length; filter to known params
            grpo_sig = inspect.signature(GRPOConfig.__init__).parameters
            grpo_extra: dict = {"num_generations": num_generations}
            if "max_completion_length" in grpo_sig:
                grpo_extra["max_completion_length"] = max_completion_length
            if "max_prompt_length" in grpo_sig:
                grpo_extra["max_prompt_length"] = max_prompt_length

            grpo_config = self._build_config(
                GRPOConfig,
                output_dir=output_dir,
                num_epochs=num_epochs,
                has_eval=False,
                save_best_model=False,
                training_kwargs=training_kwargs,
                extra_kwargs=grpo_extra,
            )

            checkpoint = self._resolve_checkpoint(output_dir, resume_from_checkpoint)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters
            trainer_kwargs: dict = {
                "model": model,
                "reward_funcs": [reward_fn],
                "args": grpo_config,
                "train_dataset": grpo_dataset,
            }
            if "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = tokenizer
            elif "processing_class" in trainer_sig:
                trainer_kwargs["processing_class"] = tokenizer

            trainer = GRPOTrainer(**trainer_kwargs)

            interrupted = False
            try:
                trainer.train(resume_from_checkpoint=checkpoint)
            except KeyboardInterrupt:
                interrupted = True
                self._save_on_interrupt(trainer, tokenizer, output_dir)

            if not interrupted:
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)

            self._cleanup(trainer, model)

            return {
                "success": True,
                "interrupted": interrupted,
                "model_path": output_dir,
                "base_model": model_name,
                "training_time_seconds": time.time() - start_time,
                "config": {
                    "trainer": "grpo",
                    "num_epochs": num_epochs,
                    "num_generations": num_generations,
                    "max_prompt_length": max_prompt_length,
                    "max_completion_length": max_completion_length,
                    "resumed_from": checkpoint,
                },
                "num_training_examples": len(dataset),
            }
        except Exception as e:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir,
                "base_model": model_name,
            }

    # ----------------------------------------------------------------
    # KTO training
    # ----------------------------------------------------------------

    async def train_kto_model(
        self,
        dataset: Any,
        output_dir: str,
        base_model: Optional[str] = None,
        num_epochs: int = 3,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        beta: float = 0.1,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        save_best_model: bool = True,
        evaluation_dataset: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train with KTO (Kahneman-Tversky Optimization).

        Dataset must have 'prompt', 'completion', 'label' (bool) columns.
        label=True marks a desirable completion; label=False marks undesirable.
        """
        start_time = time.time()
        model_name = base_model or self.config.base_model

        try:
            from datasets import Dataset
            from trl import KTOConfig, KTOTrainer
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or "a required dependency"
            return {
                "success": False,
                "error": f"Missing dependency: {missing}.",
                "base_model": model_name,
                "output_dir": output_dir,
            }

        try:
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)

            required = {"prompt", "completion", "label"}
            missing_cols = required - set(dataset.column_names)
            if missing_cols:
                return {
                    "success": False,
                    "error": f"KTO dataset missing columns: {missing_cols}",
                    "columns": list(dataset.column_names),
                }

            model, tokenizer = self._load_model_and_tokenizer(model_name, kwargs)
            cuda_available, bf16_supported = self._detect_precision()

            peft_config = None
            if use_lora:
                peft_config = self._build_lora_config(kwargs, lora_r, lora_alpha, lora_dropout)

            has_eval = evaluation_dataset is not None
            training_kwargs = self._pop_training_kwargs(kwargs, cuda_available, bf16_supported)
            args = self._build_config(
                KTOConfig,
                output_dir=output_dir,
                num_epochs=num_epochs,
                has_eval=has_eval,
                save_best_model=save_best_model,
                training_kwargs=training_kwargs,
                extra_kwargs={
                    "beta": beta,
                    "desirable_weight": desirable_weight,
                    "undesirable_weight": undesirable_weight,
                },
            )

            checkpoint = self._resolve_checkpoint(output_dir, resume_from_checkpoint)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            trainer_sig = inspect.signature(KTOTrainer.__init__).parameters
            trainer_kwargs: dict = {
                "model": model,
                "args": args,
                "train_dataset": dataset,
            }
            if has_eval and "eval_dataset" in trainer_sig:
                trainer_kwargs["eval_dataset"] = evaluation_dataset
            if "peft_config" in trainer_sig and peft_config is not None:
                trainer_kwargs["peft_config"] = peft_config
            if "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = tokenizer
            elif "processing_class" in trainer_sig:
                trainer_kwargs["processing_class"] = tokenizer

            trainer = KTOTrainer(**trainer_kwargs)

            interrupted = False
            try:
                trainer.train(resume_from_checkpoint=checkpoint)
            except KeyboardInterrupt:
                interrupted = True
                self._save_on_interrupt(trainer, tokenizer, output_dir)

            if not interrupted:
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)

            self._cleanup(trainer, model)

            return {
                "success": True,
                "interrupted": interrupted,
                "model_path": output_dir,
                "base_model": model_name,
                "training_time_seconds": time.time() - start_time,
                "config": {
                    "trainer": "kto",
                    "num_epochs": num_epochs,
                    "beta": beta,
                    "desirable_weight": desirable_weight,
                    "undesirable_weight": undesirable_weight,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "resumed_from": checkpoint,
                },
                "num_training_examples": len(dataset),
            }
        except Exception as e:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir,
                "base_model": model_name,
            }
