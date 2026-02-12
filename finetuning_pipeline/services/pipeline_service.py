"""
Fine-tuning Pipeline Service
============================

Service layer that wraps the fine-tuning pipeline with stateless,
JSON-serializable operations perfect for MCP exposure.
"""

import json
import torch
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time
import os
from transfer import Trainer, SFTConfig


@dataclass
class TrainingResult:
    """Result of a training operation."""
    success: bool
    model_path: str
    config: Dict[str, Any]
    training_time: float
    evaluation_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    success: bool
    prompt: str
    response: str
    generation_time: float
    tokens_per_second: float
    error: Optional[str] = None


class FineTuningService:
    """
    Service layer for fine-tuning operations.
    All methods are stateless with JSON-serializable inputs/outputs.
    """

    def __init__(self, default_base_model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize the fine-tuning service."""
        self.default_base_model = default_base_model
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.max_memory = {0: "3.5GiB", "cpu": "30GiB"}

    def clear_gpu_memory(self) -> Dict[str, Any]:
        """
        Clear GPU memory cache.
        
        Returns: Memory stats
        """
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        stats = {}
        if torch.cuda.is_available():
            stats = {
                "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            }
        
        return {
            "success": True,
            "memory_stats": stats
        }

    async def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        prompt_column: str = "instruction",
        response_column: str = "response",
        rename_prompt_to: str = "prompt"
    ) -> Dict[str, Any]:
        """
        Prepare a dataset from raw data.
        
        Args:
            data: List of dicts with instruction/response pairs
            prompt_column: Column name for prompts in input data
            response_column: Column name for responses
            rename_prompt_to: What to rename the prompt column to
            
        Returns: Dataset info
        """
        try:
            # Create dataset
            dataset = Dataset.from_list(data)
            
            # Rename columns if needed
            if prompt_column != rename_prompt_to and prompt_column in dataset.column_names:
                dataset = dataset.rename_column(prompt_column, rename_prompt_to)
            
            return {
                "success": True,
                "num_examples": len(dataset),
                "columns": dataset.column_names,
                "sample": dataset[0] if len(dataset) > 0 else None,
                "dataset_object": dataset  # For internal use
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def load_dataset_from_file(
        self,
        file_path: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Load dataset from file.
        
        Args:
            file_path: Path to dataset file
            format: File format (json, jsonl, csv)
            
        Returns: Dataset info
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }

            # ---- load raw data ----
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
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}"
                }

            dataset = Dataset.from_list(data)

            # ---- combine instruction + input ----
            if {"instruction", "input"}.issubset(dataset.column_names):
                dataset = dataset.map(
                    lambda x: {
                        "prompt": f"{x['instruction']} {x['input']}".strip()
                    }
                )

            # Optional: rename output → response
            if "output" in dataset.column_names:
                dataset = dataset.rename_column("output", "response")

            # Optional: clean up
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
                "dataset_object": dataset
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
        }

    async def train_model(
        self,
        dataset: Any,  # Dataset object or list of dicts
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a model with fine-tuning.
        
        Returns: Training results
        """
        try:
            start_time = time.time()
            
            # Use default model if none specified
            model_name = base_model or self.default_base_model
            
            # Convert dataset if needed
            if isinstance(dataset, list):
                dataset = Dataset.from_list(dataset)
            
            # Set default evaluation metrics
            if evaluation_metrics is None:
                evaluation_metrics = ["perplexity", "semantic_entropy", "token_entropy"]
            
            # Create configuration
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
                **kwargs
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

    
            
            # Get evaluation results if available
            eval_results = None
            eval_path = Path(config.evaluation_results_path)
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    eval_results = json.load(f)
            
            # Cleanup
            del trainer
            self.clear_gpu_memory()
            
            end_time = time.time()
            training_time = end_time - start_time
            
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
                "num_training_examples": len(dataset)
            }
            
        except Exception as e:
            self.clear_gpu_memory()
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir
            }

    async def run_inference(
        self,
        prompts: List[str],
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on prompts.
        
        Args:
            prompts: List of prompts to generate responses for
            model_path: Path to base model
            adapter_path: Optional path to LoRA adapter
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to use sampling
            
        Returns: Inference results
        """
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=self.bnb_config,
                max_memory=self.max_memory,
                torch_dtype=torch.bfloat16,
            )
            
            # Apply adapter if provided
            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path)
            
            results = []
            
            # Process each prompt
            for prompt_text in prompts:
                start_time = time.time()
                
                # Format prompt
                messages = [{"role": "user", "content": prompt_text}]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode response
                prompt_len = model_inputs.input_ids.shape[-1]
                new_ids = generated_ids[0][prompt_len:]
                response_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                
                end_time = time.time()
                generation_time = end_time - start_time
                tokens_generated = len(new_ids)
                tps = tokens_generated / generation_time if generation_time > 0 else 0
                
                results.append({
                    "prompt": prompt_text,
                    "response": response_text,
                    "generation_time_seconds": generation_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tps
                })
            
            # Cleanup
            del model
            del tokenizer
            self.clear_gpu_memory()
            
            return {
                "success": True,
                "results": results,
                "model_path": model_path,
                "adapter_path": adapter_path,
                "num_prompts": len(prompts)
            }
            
        except Exception as e:
            self.clear_gpu_memory()
            return {
                "success": False,
                "error": str(e),
                "model_path": model_path
            }

    async def compare_models(
        self,
        prompts: List[str],
        base_model_path: str,
        finetuned_adapter_path: str,
        max_new_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Compare base model vs fine-tuned model on same prompts.
        
        Returns: Comparison results
        """
        try:
            # Run inference on base model
            base_results = await self.run_inference(
                prompts=prompts,
                model_path=base_model_path,
                adapter_path=None,
                max_new_tokens=max_new_tokens
            )
            
            if not base_results["success"]:
                return base_results
            
            # Run inference on fine-tuned model
            finetuned_results = await self.run_inference(
                prompts=prompts,
                model_path=base_model_path,
                adapter_path=finetuned_adapter_path,
                max_new_tokens=max_new_tokens
            )
            
            if not finetuned_results["success"]:
                return finetuned_results
            
            # Create comparison
            comparisons = []
            for base, finetuned in zip(base_results["results"], finetuned_results["results"]):
                comparisons.append({
                    "prompt": base["prompt"],
                    "base_response": base["response"],
                    "finetuned_response": finetuned["response"],
                    "base_time": base["generation_time_seconds"],
                    "finetuned_time": finetuned["generation_time_seconds"],
                    "base_tps": base["tokens_per_second"],
                    "finetuned_tps": finetuned["tokens_per_second"]
                })
            
            return {
                "success": True,
                "comparisons": comparisons,
                "base_model": base_model_path,
                "finetuned_adapter": finetuned_adapter_path,
                "num_prompts": len(prompts)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Returns: Model metadata
        """
        try:
            path = Path(model_path)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Model path not found: {model_path}"
                }
            
            # Check for adapter files (LoRA)
            is_adapter = (path / "adapter_config.json").exists()
            
            info = {
                "success": True,
                "path": str(path),
                "exists": True,
                "is_adapter": is_adapter,
                "files": [f.name for f in path.iterdir()]
            }
            
            # Read adapter config if available
            if is_adapter:
                with open(path / "adapter_config.json", 'r') as f:
                    info["adapter_config"] = json.load(f)
            
            # Read training args if available
            if (path / "training_args.bin").exists():
                info["has_training_args"] = True
            
            return info
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": model_path
            }

    async def list_available_base_models(
        self,
        query: str = "",
        hf_home: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List locally available Hugging Face base models.

        This function is intended for AI agents to resolve model paths,
        NOT for humans.
        """

        hf_home = hf_home or os.getenv(
            "HF_HOME",
            os.path.join(Path.home(), ".cache", "huggingface")
        )

        hub_path = Path(hf_home) / "hub"
        if not hub_path.exists():
            return {"success": True, "models": []}

        models = []

        for model_dir in hub_path.iterdir():
            if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
                continue

            model_id = model_dir.name.replace("models--", "").replace("--", "/")
            if query and query.lower() not in model_id.lower():
                continue

            snapshots = model_dir / "snapshots"
            if not snapshots.exists():
                continue

            # Take latest snapshot (HF keeps them immutable anyway)
            snapshot = sorted(snapshots.iterdir())[-1]

            models.append({
                "id": model_id,
                "model_path": str(snapshot),
                "usable_for": ["training", "inference"]
            })

        return {
            "success": True,
            "models": models
        }
        
    async def search_huggingface_models(
        self,
        query: str = "",
        task: str = "text-generation",
        sort: str = "downloads",
        limit: int = 20,
        filter_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for models on HuggingFace Hub.
        
        Args:
            query: Search query (e.g., "llama", "mistral", "phi")
            task: Task type (text-generation, text-classification, etc.)
            sort: Sort by (downloads, likes, trending, created)
            limit: Max number of results
            filter_tags: Filter by tags (e.g., ["pytorch", "safetensors"])
            
        Returns: List of available models
        """
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Search models
            models = api.list_models(
                search=query,
                task=task,
                sort=sort,
                limit=limit,
                tags=filter_tags
            )
            
            results = []
            for model in models:
                model_info = {
                    "id": model.id,
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "library": getattr(model, "library_name", None),
                    "created_at": str(model.created_at) if hasattr(model, "created_at") else None,
                }
                
                # Get model card info if available
                try:
                    card = api.model_info(model.id)
                    model_info["card_data"] = card.card_data.to_dict() if card.card_data else {}
                except:
                    pass
                
                results.append(model_info)
            
            return {
                "success": True,
                "query": query,
                "task": task,
                "models": results,
                "count": len(results)
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "huggingface_hub not installed. Run: pip install huggingface_hub"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    async def get_huggingface_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific HuggingFace model.
        
        Args:
            model_id: Model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")
            
        Returns: Detailed model information
        """
        try:
            from huggingface_hub import HfApi, model_info
            
            api = HfApi()
            info = model_info(model_id)
            
            result = {
                "success": True,
                "model_id": model_id,
                "author": info.author,
                "sha": info.sha,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "created_at": str(info.created_at) if hasattr(info, "created_at") else None,
                "last_modified": str(info.last_modified) if hasattr(info, "last_modified") else None,
            }
            
            # Get siblings (files in the model)
            if hasattr(info, "siblings") and info.siblings:
                result["files"] = [
                    {
                        "filename": s.rfilename,
                        "size_bytes": getattr(s, "size", None)
                    }
                    for s in info.siblings[:20]  # Limit to first 20 files
                ]
            
            # Get model card data
            if info.card_data:
                card_dict = info.card_data.to_dict()
                result["card_data"] = {
                    "language": card_dict.get("language"),
                    "license": card_dict.get("license"),
                    "base_model": card_dict.get("base_model"),
                    "datasets": card_dict.get("datasets"),
                    "metrics": card_dict.get("model-index", [{}])[0].get("results", []) if card_dict.get("model-index") else []
                }
            
            return result
            
        except ImportError:
            return {
                "success": False,
                "error": "huggingface_hub not installed. Run: pip install huggingface_hub"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_id": model_id
            }

    async def search_local_models(
        self,
        query: str = "",
        hf_home: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Search locally cached Hugging Face models.

        Args:
            query: Partial model name (e.g. "llama", "mistral")
            hf_home: Optional HF cache root override
            file_extensions: Filter files (e.g. [".safetensors", ".bin"])
            limit: Max results

        Returns:
            Dict with model metadata
        """

        hf_home = hf_home or os.getenv(
            "HF_HOME",
            os.path.join(Path.home(), ".cache", "huggingface")
        )

        hub_path = Path(hf_home) / "hub"

        if not hub_path.exists():
            return {
                "success": False,
                "error": f"Hugging Face cache not found at {hub_path}"
            }

        results = []

        # models--org--repo
        model_dirs = [
            d for d in hub_path.iterdir()
            if d.is_dir() and d.name.startswith("models--")
        ]

        for model_dir in model_dirs:
            model_id = model_dir.name.replace("models--", "").replace("--", "/")

            if query and query.lower() not in model_id.lower():
                continue

            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            for snapshot in snapshots_dir.iterdir():
                files = []
                total_size = 0

                for root, _, filenames in os.walk(snapshot):
                    for fname in filenames:
                        if file_extensions and not any(
                            fname.endswith(ext) for ext in file_extensions
                        ):
                            continue

                        fpath = Path(root) / fname
                        size = fpath.stat().st_size
                        total_size += size

                        files.append({
                            "name": fname,
                            "size_mb": round(size / (1024 ** 2), 2)
                        })

                results.append({
                    "id": model_id,
                    "snapshot": snapshot.name,
                    "path": str(snapshot),
                    "files": files,
                    "total_size_mb": round(total_size / (1024 ** 2), 2)
                })

                if len(results) >= limit:
                    break

        return {
            "success": True,
            "query": query,
            "cache_dir": str(hub_path),   # ✅ ADD THIS
            "count": len(results),
            "models": results
        }


    def get_recommended_models(self, use_case: str = "general") -> Dict[str, Any]:
        """
        Get recommended models for different use cases.
        
        Args:
            use_case: Use case (general, low_memory, speed, quality, multilingual)
            
        Returns: List of recommended models
        """
        recommendations = {
            "general": [
                {
                    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                    "size": "3B parameters",
                    "memory": "~4GB GPU",
                    "description": "Good balance of quality and speed"
                },
                {
                    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                    "size": "7B parameters",
                    "memory": "~8GB GPU",
                    "description": "High quality, medium speed"
                },
                {
                    "model_id": "google/gemma-2-2b-it",
                    "size": "2B parameters",
                    "memory": "~3GB GPU",
                    "description": "Fast and efficient"
                }
            ],
            "low_memory": [
                {
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "size": "1B parameters",
                    "memory": "~2GB GPU",
                    "description": "Smallest Llama model"
                },
                {
                    "model_id": "microsoft/phi-2",
                    "size": "2.7B parameters",
                    "memory": "~3GB GPU",
                    "description": "Efficient and capable"
                },
                {
                    "model_id": "google/gemma-2-2b-it",
                    "size": "2B parameters",
                    "memory": "~3GB GPU",
                    "description": "Google's efficient model"
                }
            ],
            "speed": [
                {
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "size": "1B parameters",
                    "memory": "~2GB GPU",
                    "description": "Fastest inference"
                },
                {
                    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "size": "1.1B parameters",
                    "memory": "~2GB GPU",
                    "description": "Very fast, good for testing"
                }
            ],
            "quality": [
                {
                    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                    "size": "7B parameters",
                    "memory": "~8GB GPU",
                    "description": "Excellent quality"
                },
                {
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "size": "8B parameters",
                    "memory": "~10GB GPU",
                    "description": "Top quality from Meta"
                }
            ],
            "multilingual": [
                {
                    "model_id": "google/gemma-2-2b-it",
                    "size": "2B parameters",
                    "memory": "~3GB GPU",
                    "description": "Good multilingual support"
                },
                {
                    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                    "size": "3B parameters",
                    "memory": "~4GB GPU",
                    "description": "Supports many languages"
                }
            ],
            "indonesian": [
                {
                    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                    "size": "3B parameters",
                    "memory": "~4GB GPU",
                    "description": "Good Indonesian support"
                },
                {
                    "model_id": "google/gemma-2-2b-it",
                    "size": "2B parameters",
                    "memory": "~3GB GPU",
                    "description": "Multilingual including Indonesian"
                }
            ]
        }
        
        models = recommendations.get(use_case.lower())
        
        if models:
            return {
                "success": True,
                "use_case": use_case,
                "recommendations": models,
                "count": len(models)
            }
        else:
            available_use_cases = list(recommendations.keys())
            return {
                "success": False,
                "error": f"Unknown use case: {use_case}",
                "available_use_cases": available_use_cases
            }