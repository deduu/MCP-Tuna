"""Training operations: dataset preparation and SFT/DPO/GRPO/KTO fine-tuning.

Heavy ML dependencies are imported inside methods so non-training commands can
run without torch/transformers installed.
"""
from __future__ import annotations

import asyncio
import json
import time
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from shared.async_utils import run_sync
from shared.config import FinetuningConfig
from shared.exceptions import OOMError
from shared.multimodal_models import is_vlm_sample

from .vlm_utils import (
    build_vlm_prompt_and_images,
    get_processor_tokenizer,
    resolve_dataset_base_dir,
)
from .training_recipe_service import TrainingRecipeService


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

    @staticmethod
    def _set_global_seed(seed: Optional[int]) -> None:
        """Seed Python, NumPy, Torch, and Transformers before model/adapters are built."""
        if seed is None:
            return

        seed = int(seed)

        try:
            import random

            random.seed(seed)
        except Exception:
            pass

        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except Exception:
            pass

        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        try:
            from transformers import set_seed

            set_seed(seed)
        except Exception:
            pass

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """Resolve HF cache wrapper dirs to a concrete snapshot path when needed."""
        path = Path(model_name)
        if not path.is_dir():
            return model_name

        snapshots_dir = path / "snapshots"
        if not snapshots_dir.is_dir() or (path / "config.json").exists():
            return model_name

        snapshot_dirs = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        if not snapshot_dirs:
            return model_name

        return str(max(snapshot_dirs, key=lambda entry: entry.stat().st_mtime))

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
            "max_steps": int(kwargs.pop("max_steps", -1)),
            "weight_decay": float(kwargs.pop("weight_decay", 0.0)),
            "max_grad_norm": float(kwargs.pop("max_grad_norm", 1.0)),
            "lr_scheduler_type": str(kwargs.pop("lr_scheduler_type", "linear")),
            "warmup_ratio": float(kwargs.pop("warmup_ratio", 0.0)),
            "warmup_steps": int(kwargs.pop("warmup_steps", 0)),
            "logging_steps": int(kwargs.pop("logging_steps", 10)),
            "save_steps": int(kwargs.pop("save_steps", 200)),
            "save_total_limit": int(kwargs.pop("save_total_limit", 2)),
            "bf16": bool(kwargs.pop("bf16", bf16_supported)),
            "fp16": bool(kwargs.pop("fp16", cuda_available and not bf16_supported)),
            "report_to": report_to,
            "seed": int(kwargs.pop("seed", 42)),
            "gradient_checkpointing": bool(
                kwargs.pop("gradient_checkpointing", False)
            ),
            "optim": str(kwargs.pop("optim", "adamw_torch")),
        }

    @staticmethod
    def _is_missing_completion_error(exc: Exception) -> bool:
        """Detect TRL variants that require a ``completion`` column at runtime."""
        if isinstance(exc, KeyError) and exc.args == ("completion",):
            return True
        return str(exc).strip().strip('"').strip("'") == "completion"

    @staticmethod
    def _is_completion_only_loss_incompatible_error(exc: Exception) -> bool:
        """Detect TRL stacks that reject completion-only loss with formatted datasets."""
        message = str(exc)
        return (
            "completion_only_loss=True" in message
            and "formatting function was provided" in message
            and "incompatible" in message
        )

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

    @staticmethod
    def _cast_trainable_parameters_to_fp32(model: Any) -> int:
        """Match the notebook LoRA path by keeping trainable adapter weights in fp32."""
        try:
            import torch
        except Exception:
            return 0

        cast_count = 0
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            return 0

        for _name, param in named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            if getattr(param, "dtype", None) == torch.float32:
                cast_count += 1
                continue
            try:
                param.data = param.data.to(torch.float32)
                cast_count += 1
            except Exception:
                continue
        return cast_count

    def _apply_lora_to_model(
        self,
        model: Any,
        peft_config: Any,
        *,
        cast_trainable_fp32: bool = True,
    ) -> tuple[Any, Optional[Any], int]:
        """Prefer explicit LoRA wrapping and fall back to trainer-managed PEFT when unavailable."""
        try:
            from peft import get_peft_model
        except ImportError:
            return model, peft_config, 0

        try:
            wrapped_model = get_peft_model(model, peft_config)
        except Exception:
            return model, peft_config, 0
        cast_count = (
            self._cast_trainable_parameters_to_fp32(wrapped_model)
            if cast_trainable_fp32
            else 0
        )
        return wrapped_model, None, cast_count

    @staticmethod
    def _preflight_bnb_check() -> bool:
        """Verify bitsandbytes CUDA kernels in a subprocess.

        If the import or kernel init segfaults, only the subprocess dies
        — the MCP server stays alive.  Returns True when safe to use 4-bit.
        """
        import subprocess
        import sys

        code = (
            "import bitsandbytes; "
            "import torch; "
            "torch.cuda.is_available(); "
            "print('ok')"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False

    def _build_quantization_config(self, load_in_4bit: bool, torch_module: Any) -> Any:
        """Build a BitsAndBytes config when 4-bit loading is requested and available."""
        if not load_in_4bit:
            return None

        logger = logging.getLogger(__name__)
        if not self._preflight_bnb_check():
            logger.warning(
                "bitsandbytes preflight failed - falling back to "
                "float16/bfloat16 without 4-bit quantization"
            )
            return None

        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            compute_dtype = getattr(torch_module, "float16", torch_module.float32)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            return None

    def _load_model_and_tokenizer(
        self, model_name: str, kwargs: dict
    ) -> tuple[Any, Any]:
        """Load tokenizer + model with optional 4-bit quantization."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger = logging.getLogger(__name__)

        resolved_model_name = self._resolve_model_path(model_name)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        load_in_4bit = bool(kwargs.pop("load_in_4bit", True))
        cuda_available, bf16_supported = self._detect_precision()
        model_dtype = (
            torch.bfloat16
            if bf16_supported
            else (torch.float16 if cuda_available else torch.float32)
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_name, use_fast=True, local_files_only=local_files_only
            )
        except Exception as exc:
            message = str(exc).lower()
            fallback_markers = (
                "backend tokenizer",
                "convert a slow tokenizer to a fast one",
                "sentencepiece",
                "tiktoken",
            )
            if not any(marker in message for marker in fallback_markers):
                raise

            logger.warning(
                "Fast tokenizer unavailable for %s; falling back to slow tokenizer: %s",
                resolved_model_name,
                exc,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_name, use_fast=False, local_files_only=local_files_only
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        quantization_config = self._build_quantization_config(load_in_4bit, torch)
        if load_in_4bit and quantization_config is None:
            # Preflight: verify bitsandbytes kernels in an isolated subprocess
            # so a CUDA segfault doesn't kill the MCP server.
            if not self._preflight_bnb_check():
                logger.warning(
                    "bitsandbytes preflight failed — falling back to "
                    "float16/bfloat16 without 4-bit quantization"
                )
            else:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                except Exception:
                    quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map="auto",
            dtype=model_dtype,
            quantization_config=quantization_config,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False

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

    @staticmethod
    def _merge_system_prompts(
        default_system_prompt: Optional[str],
        row_system_prompt: Optional[str],
    ) -> Optional[str]:
        default_text = str(default_system_prompt or "").strip()
        row_text = str(row_system_prompt or "").strip()
        if default_text and row_text:
            return f"{default_text}\n\n{row_text}"
        if row_text:
            return row_text
        if default_text:
            return default_text
        return None

    @staticmethod
    def _resolve_sft_schema(
        dataset: Any,
        prompt_column: str,
        response_column: str,
    ) -> Dict[str, Any]:
        column_names = set(getattr(dataset, "column_names", []) or [])
        if prompt_column in column_names and response_column in column_names:
            return {
                "kind": "prompt_response",
                "prompt_column": prompt_column,
                "response_column": response_column,
            }
        if {"system", "user", "assistant"}.issubset(column_names):
            return {
                "kind": "chat_triplet",
                "system_column": "system",
                "user_column": "user",
                "assistant_column": "assistant",
            }
        return {"kind": "unsupported", "columns": sorted(column_names)}

    @staticmethod
    def _prepare_sft_text_dataset(
        dataset: Any,
        tokenizer: Any,
        prompt_column: str,
        response_column: str,
        system_prompt: Optional[str] = None,
        system_column: Optional[str] = None,
        user_column: Optional[str] = None,
        assistant_column: Optional[str] = None,
        text_only: bool = False,
    ) -> Any:
        """Normalize SFT rows to prompt/completion/text for TRL compatibility."""

        def to_sft_record(example: Dict[str, Any]) -> Dict[str, str]:
            if user_column and assistant_column:
                prompt = str(example.get(user_column) or "").strip()
                completion = str(example.get(assistant_column) or "").strip()
                effective_system_prompt = TrainingService._merge_system_prompts(
                    system_prompt,
                    example.get(system_column) if system_column else None,
                )
            else:
                prompt = str(example.get(prompt_column) or "").strip()
                completion = str(example.get(response_column) or "").strip()
                effective_system_prompt = system_prompt

            messages = []
            if effective_system_prompt:
                messages.append({"role": "system", "content": effective_system_prompt})
            messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ])
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prefix = f"System: {effective_system_prompt}\n" if effective_system_prompt else ""
                text = f"{prefix}User: {prompt}\nAssistant: {completion}"
            return {
                "prompt": prompt,
                "completion": completion,
                "text": text,
            }

        dataset = dataset.map(to_sft_record)
        keep_columns = {"text"} if text_only else {"prompt", "completion", "text"}
        cols_to_remove = [c for c in dataset.column_names if c not in keep_columns]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)
        return dataset

    @staticmethod
    def _register_special_tokens(
        model: Any,
        tokenizer: Any,
        special_tokens: Optional[List[str]],
    ) -> List[str]:
        if not special_tokens or not hasattr(tokenizer, "add_special_tokens"):
            return []

        unique_tokens = [
            token for token in dict.fromkeys(
                str(token).strip() for token in special_tokens if str(token).strip()
            )
        ]
        if not unique_tokens:
            return []

        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": unique_tokens}
        )
        if added and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model, "config"):
                model.config.vocab_size = len(tokenizer)
        return unique_tokens if added else []

    @staticmethod
    def _extract_latest_log_metric(
        log_history: List[Dict[str, Any]],
        key: str,
    ) -> Optional[float]:
        for entry in reversed(log_history):
            value = entry.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _summarize_trainer_metrics(
        self,
        trainer: Any,
        train_output: Any,
    ) -> Dict[str, Any]:
        """Return a compact training summary for job results and workflow outputs."""
        state = getattr(trainer, "state", None)
        raw_log_history = getattr(state, "log_history", None) or []
        log_history = [
            entry for entry in raw_log_history
            if isinstance(entry, dict)
        ]

        metrics: Dict[str, Any] = {}
        training_loss = getattr(train_output, "training_loss", None)
        if training_loss is not None:
            try:
                metrics["training_loss"] = float(training_loss)
            except (TypeError, ValueError):
                pass
        if "training_loss" not in metrics:
            latest_loss = self._extract_latest_log_metric(log_history, "loss")
            if latest_loss is not None:
                metrics["training_loss"] = latest_loss

        eval_loss = self._extract_latest_log_metric(log_history, "eval_loss")
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss

        best_metric = getattr(state, "best_metric", None)
        if best_metric is not None:
            try:
                metrics["best_eval_loss"] = float(best_metric)
            except (TypeError, ValueError):
                pass

        learning_rate = self._extract_latest_log_metric(log_history, "learning_rate")
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        grad_norm = self._extract_latest_log_metric(log_history, "grad_norm")
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        global_step = getattr(state, "global_step", None)
        if global_step is not None:
            metrics["global_step"] = int(global_step)

        epoch = getattr(state, "epoch", None)
        if epoch is not None:
            try:
                metrics["epoch"] = round(float(epoch), 2)
            except (TypeError, ValueError):
                pass

        if log_history:
            metrics["log_history_tail"] = log_history[-5:]

        return metrics

    def _load_vlm_model_and_processor(
        self, model_name: str, kwargs: dict
    ) -> tuple[Any, Any]:
        """Load a processor-backed VLM model with optional 4-bit quantization."""
        import torch
        from transformers import AutoProcessor

        logger = logging.getLogger(__name__)
        resolved_model_name = self._resolve_model_path(model_name)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        load_in_4bit = bool(kwargs.pop("load_in_4bit", True))
        cuda_available, _bf16_supported = self._detect_precision()
        model_dtype = torch.float16 if load_in_4bit and cuda_available else torch.float32

        processor = AutoProcessor.from_pretrained(
            resolved_model_name,
            local_files_only=local_files_only,
        )
        tokenizer_like = get_processor_tokenizer(processor)
        if getattr(tokenizer_like, "pad_token", None) is None and getattr(tokenizer_like, "eos_token", None) is not None:
            tokenizer_like.pad_token = tokenizer_like.eos_token

        quantization_config = self._build_quantization_config(load_in_4bit, torch)
        common_kwargs = {
            "device_map": "auto",
            "quantization_config": quantization_config,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is None:
            common_kwargs["torch_dtype"] = model_dtype

        errors: List[str] = []
        model = None
        for loader_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
            try:
                transformers_mod = __import__("transformers", fromlist=[loader_name])
                loader = getattr(transformers_mod, loader_name)
                model = loader.from_pretrained(resolved_model_name, **common_kwargs)
                break
            except Exception as exc:
                errors.append(f"{loader_name}: {exc}")

        if model is None:
            raise RuntimeError(
                "Unable to load a VLM-compatible model class. "
                + " | ".join(errors[-2:])
            )

        if quantization_config is not None:
            try:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model)
            except ImportError:
                pass

        logger.info("Loaded VLM model and processor from %s", resolved_model_name)
        return model, processor

    def _save_on_interrupt(
        self, trainer: Any, tokenizer: Any, output_dir: str
    ) -> None:
        """Best-effort checkpoint save when training is interrupted."""
        try:
            self._save_trainer_model(trainer, output_dir)
            tokenizer.save_pretrained(output_dir)
        except Exception:
            pass

    @staticmethod
    def _save_trainer_model(trainer: Any, output_dir: str) -> None:
        model = getattr(trainer, "model", None)
        save_pretrained = getattr(model, "save_pretrained", None)
        if callable(save_pretrained):
            save_pretrained(output_dir)
            return
        trainer.save_model(output_dir)

    @staticmethod
    def _save_model_artifacts(trainer: Any, tokenizer: Any, output_dir: str) -> None:
        TrainingService._save_trainer_model(trainer, output_dir)
        tokenizer.save_pretrained(output_dir)

    @staticmethod
    def _push_to_hub(model: Any, tokenizer: Any, repo_id: str) -> str:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        return f"https://huggingface.co/{repo_id}"

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
        return await run_sync(
            self._prepare_dataset_sync,
            data,
            prompt_column,
            response_column,
            rename_prompt_to,
        )

    def _prepare_dataset_sync(
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
        return await run_sync(self._load_dataset_from_file_sync, file_path, format)

    def _load_dataset_from_file_sync(
        self,
        file_path: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Load dataset from file, merge instruction+input into prompt."""
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
        recipe: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
        extra_callbacks: Optional[List] = None,
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
        original_kwargs = dict(kwargs)

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
            kwargs = TrainingRecipeService.apply_defaults(
                recipe_name=recipe,
                trainer_type="sft",
                kwargs=kwargs,
            )
            training_seed = int(kwargs.get("seed", 42))
            requested_load_in_4bit = bool(kwargs.get("load_in_4bit", True))
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)
            original_dataset = dataset
            original_evaluation_dataset = evaluation_dataset

            dataset_schema = self._resolve_sft_schema(
                dataset,
                prompt_column=prompt_column,
                response_column=response_column,
            )
            if dataset_schema["kind"] == "unsupported":
                return {
                    "success": False,
                    "error": (
                        f"Dataset must contain '{prompt_column}' and "
                        f"'{response_column}' columns or a chat-triplet schema "
                        "with 'system', 'user', and 'assistant' columns"
                    ),
                    "columns": dataset_schema["columns"],
                }

            self._set_global_seed(training_seed)
            # Load eval dataset from file if provided (and no in-memory eval dataset)
            if evaluation_dataset is None and eval_file_path:
                eval_result = await self.load_dataset_from_file(
                    eval_file_path,
                    format="jsonl" if eval_file_path.endswith(".jsonl") else "json",
                )
                if eval_result.get("success"):
                    evaluation_dataset = eval_result["dataset_object"]

            model, tokenizer = await asyncio.to_thread(
                self._load_model_and_tokenizer, model_name, kwargs
            )
            cuda_available, bf16_supported = self._detect_precision()
            recipe_config = TrainingRecipeService.get_recipe(recipe)
            effective_special_tokens = list(special_tokens or [])
            if recipe_config:
                effective_special_tokens.extend(recipe_config.get("special_tokens", []))
            applied_special_tokens = self._register_special_tokens(
                model,
                tokenizer,
                effective_special_tokens,
            )
            system_prompt = (
                recipe_config.get("system_prompt")
                if isinstance(recipe_config, dict)
                else None
            )
            use_notebook_text_only_dataset = (
                dataset_schema["kind"] == "chat_triplet" and not completion_only_loss
            )

            dataset = self._prepare_sft_text_dataset(
                dataset=dataset,
                tokenizer=tokenizer,
                prompt_column=prompt_column,
                response_column=response_column,
                system_prompt=system_prompt,
                system_column=dataset_schema.get("system_column"),
                user_column=dataset_schema.get("user_column"),
                assistant_column=dataset_schema.get("assistant_column"),
                text_only=use_notebook_text_only_dataset,
            )

            # Format eval dataset the same way
            if evaluation_dataset is not None:
                if isinstance(evaluation_dataset, list):
                    evaluation_dataset = Dataset.from_list(evaluation_dataset)
                evaluation_schema = self._resolve_sft_schema(
                    evaluation_dataset,
                    prompt_column=prompt_column,
                    response_column=response_column,
                )
                if evaluation_schema["kind"] == "unsupported":
                    return {
                        "success": False,
                        "error": (
                            "Evaluation dataset must contain prompt/response columns "
                            "or a chat-triplet schema with system/user/assistant columns"
                        ),
                        "columns": evaluation_schema["columns"],
                    }
                evaluation_dataset = self._prepare_sft_text_dataset(
                    dataset=evaluation_dataset,
                    tokenizer=tokenizer,
                    prompt_column=prompt_column,
                    response_column=response_column,
                    system_prompt=system_prompt,
                    system_column=evaluation_schema.get("system_column"),
                    user_column=evaluation_schema.get("user_column"),
                    assistant_column=evaluation_schema.get("assistant_column"),
                    text_only=(
                        evaluation_schema["kind"] == "chat_triplet"
                        and not completion_only_loss
                    ),
                )

            peft_config = None
            lora_cast_count = 0
            if use_lora:
                peft_config = self._build_lora_config(kwargs, lora_r, lora_alpha, lora_dropout)
                model, peft_config, lora_cast_count = self._apply_lora_to_model(
                    model,
                    peft_config,
                )

            has_eval = enable_evaluation and evaluation_dataset is not None
            max_seq_length = int(kwargs.pop("max_seq_length", 2048))
            packing = bool(kwargs.pop("packing", False))
            if (
                use_notebook_text_only_dataset
                and requested_load_in_4bit
                and "fp16" not in kwargs
                and "bf16" not in kwargs
            ):
                kwargs["fp16"] = False
                kwargs["bf16"] = False
            training_kwargs = self._pop_training_kwargs(kwargs, cuda_available, bf16_supported)

            # Build SFTConfig-specific extra kwargs
            sft_extra: dict = {}
            sft_config_sig = inspect.signature(SFTConfig.__init__).parameters
            if "completion_only_loss" in sft_config_sig:
                sft_extra["completion_only_loss"] = completion_only_loss
            if "dataset_text_field" in sft_config_sig:
                sft_extra["dataset_text_field"] = "text"
            if "max_length" in sft_config_sig:
                sft_extra["max_length"] = max_seq_length
            elif "max_seq_length" in sft_config_sig:
                sft_extra["max_seq_length"] = max_seq_length
            if "packing" in sft_config_sig:
                sft_extra["packing"] = packing

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
            elif "formatting_func" in trainer_sig and not completion_only_loss:
                # formatting_func is incompatible with completion_only_loss;
                # dataset already has a "text" column so newer TRL auto-detects it.
                trainer_kwargs["formatting_func"] = lambda examples: examples["text"]
            if "max_seq_length" in trainer_sig:
                trainer_kwargs["max_seq_length"] = max_seq_length
            if "packing" in trainer_sig:
                trainer_kwargs["packing"] = packing
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
            if extra_callbacks:
                callbacks.extend(extra_callbacks)
            if callbacks and "callbacks" in trainer_sig:
                trainer_kwargs["callbacks"] = callbacks

            trainer = SFTTrainer(**trainer_kwargs)

            def _train_sync():
                """Run blocking training in a thread."""
                _interrupted = False
                _train_output = None
                try:
                    _train_output = trainer.train(resume_from_checkpoint=checkpoint)
                except KeyboardInterrupt:
                    _interrupted = True
                    self._save_on_interrupt(trainer, tokenizer, output_dir)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self._cleanup(trainer, model)
                        raise OOMError(
                            "CUDA OOM during training. Reduce batch_size, "
                            "max_seq_length, or use 4-bit quantization."
                        ) from exc
                    raise
                return _interrupted, _train_output

            interrupted, train_output = await asyncio.to_thread(_train_sync)

            if not interrupted:
                await asyncio.to_thread(
                    self._save_model_artifacts, trainer, tokenizer, output_dir
                )

            # Push to HuggingFace Hub if requested
            hub_url = None
            if push_to_hub and not interrupted:
                try:
                    hub_url = await asyncio.to_thread(
                        self._push_to_hub, model, tokenizer, push_to_hub
                    )
                except Exception:
                    hub_url = None

            eval_results = None
            if save_evaluation_results:
                eval_path = Path(output_dir) / "evaluation_results.json"
                if eval_path.exists():
                    with open(eval_path, "r", encoding="utf-8") as f:
                        eval_results = json.load(f)

            await asyncio.to_thread(self._cleanup, trainer, model)
            metrics = self._summarize_trainer_metrics(trainer, train_output)

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
                    "lora_trainable_fp32_tensors": lora_cast_count,
                    "completion_only_loss": completion_only_loss,
                    "dataset_format": (
                        "text_only" if use_notebook_text_only_dataset else "structured"
                    ),
                    "max_steps": training_kwargs.get("max_steps"),
                    "warmup_steps": training_kwargs.get("warmup_steps"),
                    "seed": training_kwargs.get("seed"),
                    "resumed_from": checkpoint,
                    "recipe": recipe,
                    "special_tokens": applied_special_tokens,
                },
                "metrics": metrics,
                "evaluation_results": eval_results,
                "num_training_examples": len(dataset),
            }
            if hub_url:
                result["hub_url"] = hub_url
            return result
        except Exception as e:
            if completion_only_loss and (
                self._is_missing_completion_error(e)
                or self._is_completion_only_loss_incompatible_error(e)
            ):
                logging.getLogger(__name__).warning(
                    "SFT trainer rejected completion_only_loss; retrying with "
                    "completion_only_loss disabled for compatibility"
                )
                fallback = await self.train_model(
                    dataset=original_dataset,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    prompt_column=prompt_column,
                    response_column=response_column,
                    enable_evaluation=enable_evaluation,
                    evaluation_dataset=original_evaluation_dataset,
                    eval_file_path=eval_file_path,
                    evaluation_metrics=evaluation_metrics,
                    save_evaluation_results=save_evaluation_results,
                    resume_from_checkpoint=resume_from_checkpoint,
                    save_best_model=save_best_model,
                    completion_only_loss=False,
                    early_stopping_patience=early_stopping_patience,
                    push_to_hub=push_to_hub,
                    recipe=recipe,
                    special_tokens=special_tokens,
                    extra_callbacks=extra_callbacks,
                    **original_kwargs,
                )
                warnings = list(fallback.get("warnings") or [])
                warnings.append(
                    "completion_only_loss was disabled automatically because the "
                    "installed TRL stack expected a 'completion' column."
                )
                fallback["warnings"] = warnings
                config = fallback.get("config")
                if isinstance(config, dict):
                    config["completion_only_loss_requested"] = True
                    config["completion_only_loss_effective"] = False
                return fallback
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

    async def train_vlm_model(
        self,
        dataset: Any,
        output_dir: str,
        base_model: Optional[str] = None,
        dataset_path: Optional[str] = None,
        num_epochs: int = 3,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        extra_callbacks: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train a vision-language model with multimodal SFT."""
        start_time = time.time()
        model_name = base_model or self.config.base_model

        try:
            from datasets import Dataset
            from transformers import Trainer, TrainingArguments
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or "a required dependency"
            return {
                "success": False,
                "error": (
                    f"Missing dependency: {missing}. "
                    "Install training deps (torch, transformers, peft, datasets, pillow)."
                ),
                "base_model": model_name,
                "output_dir": output_dir,
            }

        try:
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)

            if "messages" not in dataset.column_names:
                return {
                    "success": False,
                    "error": "VLM dataset must contain a 'messages' column",
                    "columns": list(getattr(dataset, "column_names", [])),
                }

            sample = dataset[0] if len(dataset) > 0 else None
            if sample is None or not is_vlm_sample(sample):
                return {
                    "success": False,
                    "error": (
                        "Dataset does not match the canonical VLM SFT schema. "
                        "Each row needs multimodal 'messages' with at least one image block "
                        "and an assistant text response."
                    ),
                    "columns": list(getattr(dataset, "column_names", [])),
                }

            model_kwargs = dict(kwargs)
            model, processor = await asyncio.to_thread(
                self._load_vlm_model_and_processor, model_name, model_kwargs
            )

            if use_lora:
                from peft import get_peft_model

                peft_config = self._build_lora_config(model_kwargs, lora_r, lora_alpha, lora_dropout)
                model = get_peft_model(model, peft_config)

            cuda_available, bf16_supported = self._detect_precision()
            training_kwargs = self._pop_training_kwargs(model_kwargs, cuda_available, bf16_supported)
            max_seq_length = int(model_kwargs.pop("max_seq_length", 2048))
            training_args = self._build_config(
                TrainingArguments,
                output_dir=output_dir,
                num_epochs=num_epochs,
                has_eval=False,
                save_best_model=False,
                training_kwargs=training_kwargs,
                extra_kwargs={"remove_unused_columns": False},
            )

            dataset_base_dir = resolve_dataset_base_dir(dataset_path)
            tokenizer_like = get_processor_tokenizer(processor)

            def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
                texts: List[str] = []
                images_payload: List[Any] = []
                has_images = False

                for example in examples:
                    prompt_text, images = build_vlm_prompt_and_images(
                        example.get("messages", []),
                        processor=processor,
                        base_dir=dataset_base_dir,
                        add_generation_prompt=False,
                    )
                    texts.append(prompt_text)
                    images_payload.append(images if len(images) != 1 else images[0])
                    has_images = has_images or bool(images)

                processor_kwargs: Dict[str, Any] = {
                    "text": texts,
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": max_seq_length,
                }
                if has_images:
                    processor_kwargs["images"] = images_payload

                model_inputs = processor(**processor_kwargs)
                input_ids = model_inputs.get("input_ids")
                if input_ids is None:
                    raise ValueError("VLM processor did not return input_ids")

                labels = input_ids.clone()
                pad_token_id = getattr(tokenizer_like, "pad_token_id", None)
                if pad_token_id is not None:
                    labels[labels == pad_token_id] = -100
                model_inputs["labels"] = labels
                return model_inputs

            trainer_kwargs: Dict[str, Any] = {
                "model": model,
                "args": training_args,
                "train_dataset": dataset,
                "data_collator": collate_fn,
            }
            if extra_callbacks:
                trainer_kwargs["callbacks"] = extra_callbacks

            trainer = Trainer(**trainer_kwargs)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            def _train_sync() -> bool:
                interrupted = False
                try:
                    trainer.train()
                except KeyboardInterrupt:
                    interrupted = True
                    trainer.save_model(output_dir)
                    processor.save_pretrained(output_dir)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self._cleanup(trainer, model)
                        raise OOMError(
                            "CUDA OOM during VLM training. Reduce batch_size, "
                            "image count, or max_seq_length."
                        ) from exc
                    raise
                return interrupted

            interrupted = await asyncio.to_thread(_train_sync)
            if not interrupted:
                await asyncio.to_thread(self._save_model_artifacts, trainer, processor, output_dir)

            await asyncio.to_thread(self._cleanup, trainer, model)

            return {
                "success": True,
                "interrupted": interrupted,
                "model_path": output_dir,
                "base_model": model_name,
                "training_time_seconds": time.time() - start_time,
                "config": {
                    "trainer": "vlm_sft",
                    "num_epochs": num_epochs,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "dataset_path": dataset_path,
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
        recipe: Optional[str] = None,
        extra_callbacks: Optional[List] = None,
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
            kwargs = TrainingRecipeService.apply_defaults(
                recipe_name=recipe,
                trainer_type="dpo",
                kwargs=kwargs,
            )
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

            model, tokenizer = await asyncio.to_thread(
                self._load_model_and_tokenizer, model_name, kwargs
            )
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
            if extra_callbacks and "callbacks" in trainer_sig:
                trainer_kwargs["callbacks"] = list(extra_callbacks)

            trainer = DPOTrainer(**trainer_kwargs)

            def _train_sync():
                _interrupted = False
                try:
                    trainer.train(resume_from_checkpoint=checkpoint)
                except KeyboardInterrupt:
                    _interrupted = True
                    self._save_on_interrupt(trainer, tokenizer, output_dir)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self._cleanup(trainer, model)
                        raise OOMError(
                            "CUDA OOM during training. Reduce batch_size, "
                            "max_seq_length, or use 4-bit quantization."
                        ) from exc
                    raise
                return _interrupted

            interrupted = await asyncio.to_thread(_train_sync)

            if not interrupted:
                await asyncio.to_thread(
                    self._save_model_artifacts, trainer, tokenizer, output_dir
                )

            await asyncio.to_thread(self._cleanup, trainer, model)

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
                    "recipe": recipe,
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
        extra_callbacks: Optional[List] = None,
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

            model, tokenizer = await asyncio.to_thread(
                self._load_model_and_tokenizer, model_name, kwargs
            )
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
            if extra_callbacks and "callbacks" in trainer_sig:
                trainer_kwargs["callbacks"] = list(extra_callbacks)

            trainer = GRPOTrainer(**trainer_kwargs)

            def _train_sync():
                _interrupted = False
                try:
                    trainer.train(resume_from_checkpoint=checkpoint)
                except KeyboardInterrupt:
                    _interrupted = True
                    self._save_on_interrupt(trainer, tokenizer, output_dir)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self._cleanup(trainer, model)
                        raise OOMError(
                            "CUDA OOM during training. Reduce batch_size, "
                            "max_seq_length, or use 4-bit quantization."
                        ) from exc
                    raise
                return _interrupted

            interrupted = await asyncio.to_thread(_train_sync)

            if not interrupted:
                await asyncio.to_thread(
                    self._save_model_artifacts, trainer, tokenizer, output_dir
                )

            await asyncio.to_thread(self._cleanup, trainer, model)

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
        extra_callbacks: Optional[List] = None,
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

            model, tokenizer = await asyncio.to_thread(
                self._load_model_and_tokenizer, model_name, kwargs
            )
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
            if extra_callbacks and "callbacks" in trainer_sig:
                trainer_kwargs["callbacks"] = list(extra_callbacks)

            trainer = KTOTrainer(**trainer_kwargs)

            def _train_sync():
                _interrupted = False
                try:
                    trainer.train(resume_from_checkpoint=checkpoint)
                except KeyboardInterrupt:
                    _interrupted = True
                    self._save_on_interrupt(trainer, tokenizer, output_dir)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self._cleanup(trainer, model)
                        raise OOMError(
                            "CUDA OOM during training. Reduce batch_size, "
                            "max_seq_length, or use 4-bit quantization."
                        ) from exc
                    raise
                return _interrupted

            interrupted = await asyncio.to_thread(_train_sync)

            if not interrupted:
                await asyncio.to_thread(
                    self._save_model_artifacts, trainer, tokenizer, output_dir
                )

            await asyncio.to_thread(self._cleanup, trainer, model)

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
