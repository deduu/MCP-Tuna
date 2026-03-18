"""HuggingFace model search, local cache discovery, and recommendations."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.async_utils import run_sync
from shared.model_capabilities import build_model_capabilities, infer_model_modality


class ModelDiscoveryService:
    """Discovers models from HuggingFace Hub and local cache."""

    async def search_huggingface_models(
        self,
        query: str = "",
        task: str = "text-generation",
        sort: str = "downloads",
        limit: int = 20,
        filter_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search for models on HuggingFace Hub."""
        return await run_sync(
            self._search_huggingface_models_sync,
            query, task, sort, limit, filter_tags,
        )

    def _search_huggingface_models_sync(
        self,
        query: str = "",
        task: str = "text-generation",
        sort: str = "downloads",
        limit: int = 20,
        filter_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            kwargs: Dict[str, Any] = {
                "search": query,
                "pipeline_tag": task,
                "sort": sort,
                "limit": limit,
            }
            if filter_tags:
                kwargs["filter"] = filter_tags
            models = api.list_models(**kwargs)

            results = []
            for model in models:
                results.append({
                    "id": model.id,
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "library": getattr(model, "library_name", None),
                    "created_at": str(model.created_at) if hasattr(model, "created_at") else None,
                })

            return {
                "success": True,
                "query": query,
                "task": task,
                "models": results,
                "count": len(results),
            }
        except ImportError:
            return {"success": False, "error": "huggingface_hub not installed. Run: pip install huggingface_hub"}
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    async def get_huggingface_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific HuggingFace model."""
        return await run_sync(self._get_huggingface_model_info_sync, model_id)

    def _get_huggingface_model_info_sync(self, model_id: str) -> Dict[str, Any]:
        try:
            from huggingface_hub import model_info

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

            if hasattr(info, "siblings") and info.siblings:
                result["files"] = [
                    {"filename": s.rfilename, "size_bytes": getattr(s, "size", None)}
                    for s in info.siblings[:20]
                ]

            if info.card_data:
                card_dict = info.card_data.to_dict()
                result["card_data"] = {
                    "language": card_dict.get("language"),
                    "license": card_dict.get("license"),
                    "base_model": card_dict.get("base_model"),
                    "datasets": card_dict.get("datasets"),
                    "metrics": card_dict.get("model-index", [{}])[0].get("results", [])
                    if card_dict.get("model-index") else [],
                }

            return result
        except ImportError:
            return {"success": False, "error": "huggingface_hub not installed. Run: pip install huggingface_hub"}
        except Exception as e:
            return {"success": False, "error": str(e), "model_id": model_id}

    async def list_available_base_models(
        self,
        query: str = "",
        hf_home: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List locally available HuggingFace base models."""
        return await run_sync(self._list_available_base_models_sync, query, hf_home)

    def _list_available_base_models_sync(
        self,
        query: str = "",
        hf_home: Optional[str] = None,
    ) -> Dict[str, Any]:
        hf_home = hf_home or os.getenv(
            "HF_HOME", os.path.join(Path.home(), ".cache", "huggingface"),
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
            snapshot = sorted(snapshots.iterdir())[-1]
            modality = infer_model_modality(model_id, file_names=[f.name for f in snapshot.iterdir()])
            capabilities = build_model_capabilities(modality)
            models.append({
                "id": model_id,
                "model_path": str(snapshot),
                **capabilities,
            })

        return {"success": True, "models": models}

    async def search_local_models(
        self,
        query: str = "",
        hf_home: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Search locally cached HuggingFace models with file details."""
        return await run_sync(
            self._search_local_models_sync,
            query, hf_home, file_extensions, limit,
        )

    def _search_local_models_sync(
        self,
        query: str = "",
        hf_home: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        hf_home = hf_home or os.getenv(
            "HF_HOME", os.path.join(Path.home(), ".cache", "huggingface"),
        )
        hub_path = Path(hf_home) / "hub"
        if not hub_path.exists():
            return {"success": False, "error": f"HuggingFace cache not found at {hub_path}"}

        results = []
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
                        files.append({"name": fname, "size_mb": round(size / (1024**2), 2)})

                modality = infer_model_modality(
                    model_id,
                    file_names=[file_info["name"] for file_info in files],
                )
                capabilities = build_model_capabilities(modality)
                results.append({
                    "id": model_id,
                    "snapshot": snapshot.name,
                    "path": str(snapshot),
                    "model_path": str(snapshot),
                    "files": files,
                    "total_size_mb": round(total_size / (1024**2), 2),
                    **capabilities,
                })

                if len(results) >= limit:
                    break

        return {
            "success": True,
            "query": query,
            "cache_dir": str(hub_path),
            "count": len(results),
            "models": results,
        }

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get info about a local model directory."""
        try:
            path = Path(model_path)
            if not path.exists():
                return {"success": False, "error": f"Model path not found: {model_path}"}

            is_adapter = (path / "adapter_config.json").exists()
            config = None
            config_path = path / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception:
                    config = None
            file_names = [f.name for f in path.iterdir()]
            modality = infer_model_modality(str(path), config=config, file_names=file_names)
            capabilities = build_model_capabilities(modality)
            info = {
                "success": True,
                "path": str(path),
                "exists": True,
                "is_adapter": is_adapter,
                "files": file_names,
                **capabilities,
            }

            if is_adapter:
                with open(path / "adapter_config.json", "r", encoding="utf-8") as f:
                    info["adapter_config"] = json.load(f)

            if config is not None:
                info["config"] = {
                    "model_type": config.get("model_type"),
                    "architectures": config.get("architectures"),
                }
            if (path / "training_args.bin").exists():
                info["has_training_args"] = True

            return info
        except Exception as e:
            return {"success": False, "error": str(e), "path": model_path}

    def get_recommended_models(self, use_case: str = "general") -> Dict[str, Any]:
        """Get recommended models for different use cases."""
        recommendations = {
            "general": [
                {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "size": "3B parameters", "memory": "~4GB GPU", "description": "Good balance of quality and speed"},
                {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B parameters", "memory": "~8GB GPU", "description": "High quality, medium speed"},
                {"model_id": "google/gemma-2-2b-it", "size": "2B parameters", "memory": "~3GB GPU", "description": "Fast and efficient"},
            ],
            "low_memory": [
                {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "size": "1B parameters", "memory": "~2GB GPU", "description": "Smallest Llama model"},
                {"model_id": "microsoft/phi-2", "size": "2.7B parameters", "memory": "~3GB GPU", "description": "Efficient and capable"},
                {"model_id": "google/gemma-2-2b-it", "size": "2B parameters", "memory": "~3GB GPU", "description": "Google's efficient model"},
            ],
            "speed": [
                {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "size": "1B parameters", "memory": "~2GB GPU", "description": "Fastest inference"},
                {"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size": "1.1B parameters", "memory": "~2GB GPU", "description": "Very fast, good for testing"},
            ],
            "quality": [
                {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B parameters", "memory": "~8GB GPU", "description": "Excellent quality"},
                {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "size": "8B parameters", "memory": "~10GB GPU", "description": "Top quality from Meta"},
            ],
            "multilingual": [
                {"model_id": "google/gemma-2-2b-it", "size": "2B parameters", "memory": "~3GB GPU", "description": "Good multilingual support"},
                {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "size": "3B parameters", "memory": "~4GB GPU", "description": "Supports many languages"},
            ],
            "indonesian": [
                {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "size": "3B parameters", "memory": "~4GB GPU", "description": "Good Indonesian support"},
                {"model_id": "google/gemma-2-2b-it", "size": "2B parameters", "memory": "~3GB GPU", "description": "Multilingual including Indonesian"},
            ],
        }

        models = recommendations.get(use_case.lower())
        if models:
            return {"success": True, "use_case": use_case, "recommendations": models, "count": len(models)}
        return {"success": False, "error": f"Unknown use case: {use_case}", "available_use_cases": list(recommendations.keys())}
