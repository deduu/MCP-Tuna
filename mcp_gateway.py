"""
MCP Tuna Unified Gateway
====================================

Single entry point exposing all pipeline operations as MCP tools.
Uses agentsoul's MCPServer (production-grade HTTP+stdio transport).

85 tools across 17 namespaces:
  system, file, extract, generate, clean, normalize, evaluate,
  evaluate_model, dataset, finetune, test, validate, host, workflow,
  orchestration, judge, ft_eval
"""
from __future__ import annotations

import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from agentsoul.core.agent import AgentSoul
from agentsoul.tools.service import ToolService
from agentsoul.server import MCPServer
from app.core.config import settings
from dotenv import load_dotenv
from shared.async_utils import call_maybe_async
from shared.config import (
    AdvancedJudgeConfig,
    ChatConfig,
    FTEvaluatorConfig,
    GeneratorConfig,
    CleaningConfig,
    NormalizationConfig,
    EvaluatorConfig,
    HostingConfig,
    OrchestrationConfig,
    ModelEvaluationConfig,
)
from shared.persistence import get_persistence_service

TechniqueName = Literal["sft", "dpo", "grpo", "kto"]
SchemaTechniqueName = Literal["sft", "dpo", "grpo", "kto", "vlm_sft"]
DifficultyOrder = Literal["easy_first", "hard_first"]
ResourceQuantization = Literal["4bit", "8bit", "none", "fp16", "bf16", "fp32"]
GGUFQuantization = Literal["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"]
DatasetFormat = Literal["jsonl", "json", "csv", "parquet"]
DatasetSaveFormat = Literal["jsonl", "json", "parquet"]
ModelUseCase = Literal[
    "general",
    "low_memory",
    "speed",
    "quality",
    "multilingual",
    "indonesian",
]
OrchestrationTrainingFormat = Literal["sft", "dpo", "grpo"]
ResultExportFormat = Literal["jsonl", "json"]
ModelEvalExportFormat = Literal["jsonl", "json", "xlsx"]


class TunaGateway:
    """Unified MCP gateway that composes all MCP Tuna pipeline services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        load_dotenv(override=False)
        config = config or {}
        self.mcp = MCPServer("mcp-tuna-gateway", "1.0.0")
        self.mcp.configure_http_app = self._configure_http_app

        # Lazily-initialized services (avoid heavy imports at gateway startup)
        self._generator_svc = None
        self._cleaning_svc = None
        self._normalization_svc = None
        self._evaluator_svc = None
        self._model_evaluator_svc = None
        self._finetuning_svc = None
        self._hosting_svc = None
        self._orchestrator = None
        self._orchestration_data_svc = None
        self._advanced_judge_svc = None
        self._ft_evaluator_svc = None
        self._job_manager_instance = None
        self._workflow_job_manager_instance = None
        self._dataset_svc = None
        self._file_svc = None
        self._chat_sessions: Dict[str, Any] = {}
        self._persistence = get_persistence_service()

        self._config = config
        self._register_all_tools()
        self._wrap_tools_with_diagnostics()

    @staticmethod
    def _format_sse(event: str, payload: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _deployment_endpoint(deployment: Dict[str, Any]) -> Optional[str]:
        deployment_host = deployment.get("host")
        if deployment_host in {"0.0.0.0", "::"}:
            deployment_host = "127.0.0.1"
        if deployment.get("transport") != "http" or not deployment_host:
            return None
        return f"http://{deployment_host}:{deployment['port']}"

    @staticmethod
    def _conversation_title(content: Any) -> Optional[str]:
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"].strip())
            text = " ".join(part for part in parts if part).strip()
        else:
            text = ""

        if not text:
            return None

        single_line = " ".join(text.split())
        if len(single_line) <= 72:
            return single_line
        return f"{single_line[:69].rstrip()}..."

    async def _persist_conversation_metadata(
        self,
        *,
        conversation_id: str,
        deployment_id: Optional[str],
        modality: str,
        endpoint: Optional[str],
        model_path: Optional[str],
        adapter_path: Optional[str],
        system_prompt: Optional[str],
    ) -> None:
        await self._persistence.upsert_conversation(
            {
                "conversation_id": conversation_id,
                "deployment_id": deployment_id,
                "modality": modality,
                "endpoint": endpoint,
                "model_path": model_path,
                "adapter_path": adapter_path,
                "system_prompt": system_prompt,
            }
        )

    async def _persist_conversation_exchange(
        self,
        *,
        conversation_id: str,
        user_content: Any,
        assistant_content: Any,
    ) -> None:
        persisted = await self._persistence.get_conversation(conversation_id)
        if persisted and not (persisted.get("title") or persisted.get("metadata", {}).get("title")):
            title = self._conversation_title(user_content)
            if title:
                metadata = dict(persisted.get("metadata") or {})
                metadata["title"] = title
                await self._persistence.upsert_conversation(
                    {
                        "conversation_id": conversation_id,
                        "deployment_id": persisted.get("deployment_id"),
                        "modality": persisted.get("modality"),
                        "endpoint": persisted.get("endpoint"),
                        "model_path": persisted.get("model_path"),
                        "adapter_path": persisted.get("adapter_path"),
                        "system_prompt": persisted.get("system_prompt"),
                        "message_count": persisted.get("message_count"),
                        "metadata": metadata,
                    }
                )
        await self._persistence.append_conversation_message(
            conversation_id,
            role="user",
            content=user_content,
        )
        await self._persistence.append_conversation_message(
            conversation_id,
            role="assistant",
            content=assistant_content,
        )

    async def _restore_persisted_conversation(
        self,
        session: Any,
        conversation_id: str,
    ) -> None:
        persisted = await self._persistence.get_conversation(conversation_id)
        if not persisted:
            return
        restore_history = getattr(session, "restore_history", None)
        if callable(restore_history):
            restore_history([
                {
                    "role": message.get("role", "user"),
                    "content": message.get("content"),
                }
                for message in persisted.get("messages", [])
            ])

    async def _get_or_create_text_chat_session(
        self,
        *,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        conversation_id: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        system_prompt: Optional[str] = None,
        prefer_runtime_metrics: bool = False,
    ) -> Dict[str, Any]:
        from hosting_pipeline.services.chat_service import ChatSession

        cid = conversation_id or str(uuid.uuid4())[:8]
        if cid in self._chat_sessions:
            session = self._chat_sessions[cid]
            if session.get_info().get("modality") != "text":
                return {
                    "success": False,
                    "error": "Conversation is multimodal. Use host.chat_vlm for this conversation.",
                    "conversation_id": cid,
                }
            if hasattr(session, "update_generation_config"):
                session.update_generation_config(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            return {
                "success": True,
                "conversation_id": cid,
                "deployment_id": deployment_id,
                "session": session,
            }

        provider = None
        inference_service = None
        resolved_endpoint = endpoint
        resolved_model_path = model_path
        resolved_adapter_path = adapter_path
        resolved_api_path = "/generate"
        modality = "text"
        persisted_conversation = None

        if conversation_id:
            persisted_conversation = await self._persistence.get_conversation(cid)
            if persisted_conversation:
                deployment_id = deployment_id or persisted_conversation.get("deployment_id")
                resolved_endpoint = resolved_endpoint or persisted_conversation.get("endpoint")
                resolved_model_path = resolved_model_path or persisted_conversation.get("model_path")
                resolved_adapter_path = (
                    resolved_adapter_path or persisted_conversation.get("adapter_path")
                )
                system_prompt = system_prompt or persisted_conversation.get("system_prompt")
                modality = persisted_conversation.get("modality") or modality

        if deployment_id:
            deployment = self.hoster.get_deployment(deployment_id)
            if deployment is None:
                deployment = await self._persistence.get_deployment(deployment_id)
                if deployment is None:
                    return {
                        "success": False,
                        "error": f"Deployment {deployment_id} not found",
                    }

            deployment_endpoint = self._deployment_endpoint(deployment)
            modality = deployment.get("modality", "text")
            if modality != "text":
                return {
                    "success": False,
                    "error": "Deployment is vision-language. Use host.chat_vlm instead.",
                }

            if deployment.get("type") == "api" and deployment_endpoint and not prefer_runtime_metrics:
                resolved_endpoint = deployment_endpoint
                resolved_api_path = deployment.get("api_path") or "/generate"
            else:
                resolved_model_path = deployment.get("model_path")
                resolved_adapter_path = deployment.get("adapter_path")
                provider = deployment.get("provider")
                inference_service = deployment.get("inference_service")

        config = ChatConfig(
            endpoint=resolved_endpoint,
            model_path=resolved_model_path,
            adapter_path=resolved_adapter_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system_prompt=system_prompt,
            modality=modality,
            api_path=resolved_api_path,
        )
        session = ChatSession(
            config,
            provider=provider,
            inference_service=inference_service,
        )
        await session.initialize()
        if persisted_conversation:
            await self._restore_persisted_conversation(session, cid)
        self._chat_sessions[cid] = session
        await self._persist_conversation_metadata(
            conversation_id=cid,
            deployment_id=deployment_id,
            modality=modality,
            endpoint=resolved_endpoint,
            model_path=resolved_model_path,
            adapter_path=resolved_adapter_path,
            system_prompt=system_prompt,
        )
        return {
            "success": True,
            "conversation_id": cid,
            "deployment_id": deployment_id,
            "session": session,
        }

    async def _configure_http_app(self, app: Any) -> None:
        from fastapi.responses import JSONResponse, StreamingResponse

        @app.post("/mcp/chat/stream")
        async def stream_host_chat(payload: Dict[str, Any]):
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip():
                return JSONResponse(
                    {"success": False, "error": "message must be a non-empty string"},
                    status_code=400,
                )

            session_result = await self._get_or_create_text_chat_session(
                deployment_id=payload.get("deployment_id"),
                endpoint=payload.get("endpoint"),
                model_path=payload.get("model_path"),
                adapter_path=payload.get("adapter_path"),
                conversation_id=payload.get("conversation_id"),
                max_new_tokens=int(payload.get("max_new_tokens") or 512),
                temperature=float(payload.get("temperature", 0.7)),
                top_p=float(payload.get("top_p", 0.95)),
                top_k=int(payload.get("top_k", 50)),
                system_prompt=payload.get("system_prompt"),
                prefer_runtime_metrics=bool(payload.get("prefer_runtime_metrics", False)),
            )
            if not session_result.get("success"):
                return JSONResponse(session_result, status_code=400)

            session = session_result["session"]
            conversation_id = session_result["conversation_id"]
            deployment_id = session_result.get("deployment_id")

            async def event_generator():
                try:
                    async for event in session.stream_message_events(message):
                        if event.get("type") == "token":
                            yield self._format_sse("token", {"content": event.get("content", "")})
                            continue

                        if event.get("type") == "complete":
                            await self._persist_conversation_exchange(
                                conversation_id=conversation_id,
                                user_content=message,
                                assistant_content=event.get("response", ""),
                            )
                            yield self._format_sse(
                                "complete",
                                {
                                    "success": True,
                                    "conversation_id": conversation_id,
                                    "deployment_id": deployment_id,
                                    "response": event.get("response", ""),
                                    "turns": session.get_info()["turns"],
                                    "metrics": event.get("metrics"),
                                    "usage": event.get("usage"),
                                    "model_id": event.get("model_id"),
                                },
                            )
                except Exception as exc:
                    yield self._format_sse("error", {"error": str(exc)})

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    # ------------------------------------------------------------------ #
    # Lazy service accessors
    # ------------------------------------------------------------------ #
    @property
    def generator(self):
        if self._generator_svc is None:
            try:
                from data_generator_pipeline.services.pipeline_service import PipelineService
            except ImportError:
                raise ImportError(
                    "Data generation tools require: pip install mcp-tuna[data]"
                ) from None
            from shared.provider_factory import create_llm
            gen_config = GeneratorConfig(**{
                k: v for k, v in self._config.get("generator", {}).items()
                if k in GeneratorConfig.model_fields
            })
            llm = create_llm(gen_config)
            self._generator_svc = PipelineService(llm, gen_config)
        return self._generator_svc

    @property
    def cleaner(self):
        if self._cleaning_svc is None:
            from data_cleaning_pipeline.services.cleaning_service import DataCleaningService
            self._cleaning_svc = DataCleaningService()
        return self._cleaning_svc

    @property
    def normalizer(self):
        if self._normalization_svc is None:
            from data_normalization_pipeline.services.normalization_service import DataNormalizationService
            self._normalization_svc = DataNormalizationService()
        return self._normalization_svc

    @property
    def evaluator(self):
        if self._evaluator_svc is None:
            try:
                from data_evaluator_pipeline.services.pipeline_service import EvaluatorService
            except ImportError as e:
                raise ImportError(
                    "Evaluation tools unavailable: "
                    f"{e}. Install with `uv sync --extra eval` or `uv sync --extra all`."
                ) from None
            eval_config = EvaluatorConfig(**self._config.get("evaluator", {}))
            self._evaluator_svc = EvaluatorService(eval_config)
        return self._evaluator_svc

    @property
    def model_evaluator(self):
        if self._model_evaluator_svc is None:
            try:
                from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService
            except ImportError:
                raise ImportError(
                    "Model evaluation tools require: pip install mcp-tuna[model-eval]"
                ) from None
            eval_config = ModelEvaluationConfig(**{
                k: v for k, v in self._config.get("model_evaluator", {}).items()
                if k in ModelEvaluationConfig.model_fields
            })
            self._model_evaluator_svc = ModelEvaluationService(eval_config)
        return self._model_evaluator_svc

    @property
    def finetuner(self):
        if self._finetuning_svc is None:
            try:
                from finetuning_pipeline.services.pipeline_service import FineTuningService
            except ImportError:
                raise ImportError(
                    "Training tools require: pip install mcp-tuna[training]"
                ) from None
            self._finetuning_svc = FineTuningService(
                default_base_model=self._config.get("finetuning", {}).get(
                    "base_model", "meta-llama/Llama-3.2-3B-Instruct"
                )
            )
        return self._finetuning_svc

    @property
    def hoster(self):
        if self._hosting_svc is None:
            try:
                from hosting_pipeline.services.hosting_service import HostingService
            except ImportError:
                raise ImportError(
                    "Hosting tools require: pip install mcp-tuna[hosting]"
                ) from None
            self._hosting_svc = HostingService()
        return self._hosting_svc

    @property
    def orchestration_data_service(self):
        if self._orchestration_data_svc is None:
            try:
                from orchestration.orchestration_trainer import OrchestrationDataService
            except ImportError:
                raise ImportError(
                    "Orchestration tools require: pip install mcp-tuna[orchestration]"
                ) from None
            from orchestration.rewards import OrchestrationRewardFunction
            from shared.provider_factory import create_llm
            orch_config = OrchestrationConfig(**{
                k: v for k, v in self._config.get("orchestration", {}).items()
                if k in OrchestrationConfig.model_fields
            })
            llm = create_llm(orch_config)
            reward_fn = OrchestrationRewardFunction(
                llm, weights=orch_config.reward_weights,
            )
            self._orchestration_data_svc = OrchestrationDataService(llm, reward_fn)
        return self._orchestration_data_svc

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from orchestration.workflow import PipelineOrchestrator
            self._orchestrator = PipelineOrchestrator(
                generator=self.generator,
                cleaner=self.cleaner,
                normalizer=self.normalizer,
                evaluator=self.evaluator,
                finetuner=self.finetuner,
                hoster=self.hoster,
                orchestration_data_service=self.orchestration_data_service,
            )
        return self._orchestrator

    @property
    def advanced_judge(self):
        if self._advanced_judge_svc is None:
            try:
                from model_evaluator_pipeline.services.judge_service import AdvancedJudgeService
            except ImportError:
                raise ImportError(
                    "Judge tools require: pip install mcp-tuna[model-eval]"
                ) from None
            judge_config = AdvancedJudgeConfig(**{
                k: v for k, v in self._config.get("advanced_judge", {}).items()
                if k in AdvancedJudgeConfig.model_fields
            })
            self._advanced_judge_svc = AdvancedJudgeService(judge_config)
        return self._advanced_judge_svc

    @property
    def ft_evaluator(self):
        if self._ft_evaluator_svc is None:
            try:
                from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
            except ImportError:
                raise ImportError(
                    "Fine-tune evaluation tools require: pip install mcp-tuna[model-eval]"
                ) from None
            ft_config = FTEvaluatorConfig(**{
                k: v for k, v in self._config.get("ft_evaluator", {}).items()
                if k in FTEvaluatorConfig.model_fields
            })
            self._ft_evaluator_svc = FTEvaluatorService(ft_config)
        return self._ft_evaluator_svc

    @property
    def job_manager(self):
        if self._job_manager_instance is None:
            from shared.training_jobs import TrainingJobManager
            self._job_manager_instance = TrainingJobManager(max_concurrent=1, namespace="training")
        return self._job_manager_instance

    @property
    def workflow_job_manager(self):
        if self._workflow_job_manager_instance is None:
            from shared.training_jobs import TrainingJobManager
            self._workflow_job_manager_instance = TrainingJobManager(max_concurrent=1, namespace="workflow")
        return self._workflow_job_manager_instance

    @property
    def dataset_service(self):
        if self._dataset_svc is None:
            from shared.dataset_service import DatasetService
            self._dataset_svc = DatasetService()
        return self._dataset_svc

    @property
    def file_service(self):
        if self._file_svc is None:
            from app.services.files.service import FileService
            self._file_svc = FileService(str(settings.files.upload_root))
        return self._file_svc

    def _configured_model_browser_roots(self) -> List[tuple[str, str, Path]]:
        roots: List[tuple[str, str, Path]] = []
        configured_paths: List[str] = []

        single_root = os.getenv("MODEL_ROOT", "").strip()
        if single_root:
            configured_paths.append(single_root)

        multi_roots = os.getenv("MODEL_BROWSE_ROOTS", "").strip()
        if multi_roots:
            configured_paths.extend(
                item.strip() for item in multi_roots.split(os.pathsep) if item.strip()
            )

        for index, configured_path in enumerate(configured_paths, start=1):
            root_path = Path(configured_path).expanduser().resolve()
            root_id = "model_root" if index == 1 else f"model_root_{index}"
            label = "Model Root" if index == 1 else f"Model Root {index}"
            roots.append((root_id, label, root_path))

        return roots

    def _deployment_browser_roots(self) -> List[Dict[str, Any]]:
        workspace_root = Path.cwd().resolve()
        output_root = (workspace_root / "output").resolve()
        uploads_root = settings.files.upload_root.resolve()
        hf_home = Path(
            os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        ).resolve()
        hf_cache_root = (hf_home / "hub").resolve()

        candidates = [
            ("workspace", "Workspace", workspace_root),
            ("output", "Output", output_root),
            ("uploads", "Uploads", uploads_root),
            ("hf_cache", "HF Cache", hf_cache_root),
            *self._configured_model_browser_roots(),
        ]

        roots: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()
        for root_id, label, root_path in candidates:
            normalized = str(root_path)
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            roots.append({
                "id": root_id,
                "label": label,
                "path": normalized,
                "exists": root_path.exists(),
            })
        return roots

    def _resolve_deployment_browser_root(self, root_id: str) -> Path:
        for root in self._deployment_browser_roots():
            if root["id"] == root_id:
                return Path(root["path"]).resolve()
        raise ValueError(f"Unknown browse root: {root_id}")

    @staticmethod
    def _resolve_hf_cache_snapshot(target_path: Path) -> Optional[Path]:
        """If target_path is an HF cache wrapper (has snapshots/ but no config.json),
        resolve to the latest snapshot directory."""
        snapshots_dir = target_path / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        # Only redirect if this looks like an HF cache wrapper (blobs/refs/snapshots)
        if (target_path / "config.json").exists():
            return None  # Already a real model directory
        # Pick the most recently modified snapshot
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            return None
        return max(snapshot_dirs, key=lambda d: d.stat().st_mtime)

    def _browse_deployment_directory(
        self,
        root_id: str,
        path: str = ".",
    ) -> Dict[str, Any]:
        try:
            root_path = self._resolve_deployment_browser_root(root_id)
            relative_path = path or "."
            target_path = (root_path / relative_path).resolve()
            if root_path != target_path and root_path not in target_path.parents:
                return {"success": False, "error": f"Path escapes root '{root_id}': {path}"}
            if not target_path.exists():
                return {"success": False, "error": f"Directory not found: {target_path}"}
            if not target_path.is_dir():
                return {"success": False, "error": f"Not a directory: {target_path}"}

            # Auto-resolve HF cache wrapper directories to the latest snapshot
            resolved_snapshot = self._resolve_hf_cache_snapshot(target_path)
            if resolved_snapshot is not None:
                # Ensure the resolved snapshot is still within the root
                if root_path == resolved_snapshot or root_path in resolved_snapshot.parents:
                    target_path = resolved_snapshot

            entries = []
            for entry in sorted(
                target_path.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            ):
                rel_path = entry.relative_to(root_path)
                entries.append({
                    "name": entry.name,
                    "path": str(rel_path).replace("\\", "/"),
                    "absolute_path": str(entry),
                    "type": "directory" if entry.is_dir() else "file",
                    "selectable": entry.is_dir(),
                })

            current_rel = "." if target_path == root_path else str(
                target_path.relative_to(root_path)
            ).replace("\\", "/")
            parent_rel = None
            if target_path != root_path:
                parent = target_path.parent
                parent_rel = (
                    "."
                    if parent == root_path
                    else str(parent.relative_to(root_path)).replace("\\", "/")
                )

            return {
                "success": True,
                "root_id": root_id,
                "root_path": str(root_path),
                "current_path": current_rel,
                "current_absolute_path": str(target_path),
                "parent_path": parent_rel,
                "entries": entries,
                "count": len(entries),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "root_id": root_id, "path": path}

    async def _invoke_dependency(self, func, /, *args, **kwargs):
        """Gateway boundary: await async services and offload sync services."""
        return await call_maybe_async(func, *args, **kwargs)

    @staticmethod
    def _coerce_tool_result(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _orchestration_tool_name_allowed(tool_name: str) -> bool:
        blocked_prefixes = ("orchestration.", "workflow.")
        blocked_names = {
            "host.stop",
            "host.list_deployments",
            "host.health",
        }
        return not tool_name.startswith(blocked_prefixes) and tool_name not in blocked_names

    async def _execute_internal_gateway_tool(self, tool_name: str, **kwargs) -> Any:
        tool_info = self.mcp._tools.get(tool_name)
        if tool_info is None:
            raise ValueError(f"Unknown gateway tool: {tool_name}")
        result = await call_maybe_async(tool_info["func"], **kwargs)
        return self._coerce_tool_result(result)

    def _build_internal_orchestration_tool_service(self) -> ToolService:
        tool_service = ToolService()
        for tool_name, tool_info in self.mcp._tools.items():
            if not self._orchestration_tool_name_allowed(tool_name):
                continue

            async def internal_wrapper(_tool_name: str = tool_name, **kwargs) -> Any:
                return await self._execute_internal_gateway_tool(_tool_name, **kwargs)

            tool_service.register_tool(
                name=tool_name,
                func=internal_wrapper,
                description={
                    "name": tool_name,
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("schema", {}),
                },
            )
        return tool_service

    def _build_internal_orchestration_agent(self) -> AgentSoul:
        from shared.provider_factory import create_llm
        from shared.config import PipelineConfig

        llm = create_llm(PipelineConfig())
        return AgentSoul(
            llm_provider=llm,
            tool_service=self._build_internal_orchestration_tool_service(),
            system_prompt=(
                "You are an orchestration agent. Use the available tools to solve tasks "
                "efficiently and accurately. Prefer multi-step plans only when needed."
            ),
            max_turns=8,
        )

    def _default_orchestration_tool_descriptions(self) -> List[Dict[str, Any]]:
        return self._build_internal_orchestration_tool_service().get_tool_descriptions()

    # ------------------------------------------------------------------ #
    # Auto-deploy helper
    # ------------------------------------------------------------------ #
    async def _auto_deploy_if_requested(
        self,
        train_result: Dict[str, Any],
        deploy: bool,
        deploy_port: int,
        base_model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Conditionally deploy a trained model after successful training.

        Returns the deployment result dict if deployed, None otherwise.
        """
        if not deploy or not train_result.get("success"):
            return None

        model_path = self._training_output_path(train_result)
        if not model_path:
            return None

        config = self._build_hosting_config_for_training_result(
            train_result=train_result,
            deploy_port=deploy_port,
            base_model=base_model,
        )
        if config is None:
            return None
        deployer = (
            self.hoster.deploy_vlm_as_mcp
            if config.modality == "vision-language"
            else self.hoster.deploy_as_mcp
        )
        result = await deployer(config)
        if result.get("success") and config.modality == "text":
            endpoint = result.get("endpoint", "")
            result["chat_command"] = (
                f"python scripts/chat_cli.py --endpoint {endpoint}"
            )
        return result

    @staticmethod
    def _training_output_path(train_result: Dict[str, Any]) -> Optional[str]:
        return train_result.get("model_path") or train_result.get("final_model_path")

    @classmethod
    def _training_uses_adapter(cls, train_result: Dict[str, Any]) -> bool:
        config = train_result.get("config")
        if isinstance(config, dict):
            use_lora = config.get("use_lora")
            if isinstance(use_lora, bool):
                return use_lora
            if config.get("trainer") == "grpo":
                return False

        stage_results = train_result.get("stage_results")
        if isinstance(stage_results, list) and stage_results:
            last_stage = stage_results[-1]
            if isinstance(last_stage, dict):
                nested = last_stage.get("training_result")
                if isinstance(nested, dict):
                    return cls._training_uses_adapter(nested)

        return True

    @staticmethod
    def _training_modality(train_result: Dict[str, Any]) -> str:
        config = train_result.get("config")
        if isinstance(config, dict) and config.get("trainer") == "vlm_sft":
            return "vision-language"
        return "text"

    def _build_hosting_config_for_training_result(
        self,
        *,
        train_result: Dict[str, Any],
        deploy_port: int,
        base_model: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> Optional[HostingConfig]:
        model_path = self._training_output_path(train_result)
        if not model_path:
            return None

        if self._training_uses_adapter(train_result):
            resolved_base = base_model or self.finetuner.config.base_model
            return HostingConfig(
                model_path=resolved_base,
                adapter_path=model_path,
                port=deploy_port,
                quantization=quantization,
                modality=self._training_modality(train_result),
            )

        return HostingConfig(
            model_path=model_path,
            adapter_path=None,
            port=deploy_port,
            quantization=quantization,
            modality=self._training_modality(train_result),
        )

    @staticmethod
    def _detect_dataset_format(dataset_path: str) -> str:
        suffix = Path(dataset_path).suffix.lower()
        if suffix == ".json":
            return "json"
        if suffix == ".csv":
            return "csv"
        if suffix == ".parquet":
            return "parquet"
        return "jsonl"

    async def _run_workflow_training_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        extra_callbacks: Optional[List[Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if tool_name == "finetune.sequential_train":
            stages = params["stages"]
            if isinstance(stages, str):
                try:
                    stages = json.loads(stages)
                except json.JSONDecodeError as exc:
                    return {"success": False, "error": f"Invalid stages JSON: {exc}"}
            result = await self.finetuner.train_sequential(
                stages=stages,
                output_dir=params["output_dir"],
                base_model=params.get("base_model"),
                merge_between_stages=params.get("merge_between_stages", True),
                extra_callbacks=extra_callbacks,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result,
                params.get("deploy", False),
                params.get("deploy_port", 8001),
                params.get("base_model"),
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return result

        training_tools = {
            "finetune.train",
            "finetune.train_dpo",
            "finetune.train_grpo",
            "finetune.train_kto",
            "finetune.train_curriculum",
        }
        if tool_name not in training_tools:
            return None

        dataset_path = params["dataset_path"]
        load_result = await self.finetuner.load_dataset_from_file(
            dataset_path,
            self._detect_dataset_format(dataset_path),
        )
        if not load_result.get("success"):
            return load_result

        dataset_obj = load_result["dataset_object"]
        if tool_name == "finetune.train":
            result = await self.finetuner.train_model(
                dataset=dataset_obj,
                output_dir=params["output_dir"],
                base_model=params.get("base_model"),
                num_epochs=params.get("num_epochs", 3),
                use_lora=params.get("use_lora", True),
                lora_r=params.get("lora_r", 8),
                lora_alpha=params.get("lora_alpha", 16),
                lora_dropout=params.get("lora_dropout", 0.05),
                completion_only_loss=params.get("completion_only_loss", True),
                early_stopping_patience=params.get("early_stopping_patience"),
                eval_file_path=params.get("eval_file_path"),
                push_to_hub=params.get("push_to_hub"),
                lr_scheduler_type=params.get("lr_scheduler_type", "linear"),
                warmup_ratio=params.get("warmup_ratio", 0.0),
                weight_decay=params.get("weight_decay", 0.0),
                max_grad_norm=params.get("max_grad_norm", 1.0),
                learning_rate=params.get("learning_rate", 2e-4),
                report_to=params.get("report_to") or [],
                max_seq_length=params.get("max_seq_length", 2048),
                per_device_train_batch_size=params.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
                gradient_checkpointing=params.get("gradient_checkpointing", False),
                optim=params.get("optim", "adamw_torch"),
                load_in_4bit=params.get("load_in_4bit", True),
                extra_callbacks=extra_callbacks,
            )
        elif tool_name == "finetune.train_dpo":
            result = await self.finetuner.train_dpo_model(
                dataset=dataset_obj,
                output_dir=params["output_dir"],
                base_model=params.get("base_model"),
                num_epochs=params.get("num_epochs", 3),
                beta=params.get("beta", 0.1),
                use_lora=params.get("use_lora", True),
                lora_r=params.get("lora_r", 8),
                resume_from_checkpoint=params.get("resume_from_checkpoint"),
                load_in_4bit=params.get("load_in_4bit", True),
                extra_callbacks=extra_callbacks,
            )
        elif tool_name == "finetune.train_grpo":
            result = await self.finetuner.train_grpo_model(
                dataset=dataset_obj,
                output_dir=params["output_dir"],
                base_model=params.get("base_model"),
                num_epochs=params.get("num_epochs", 3),
                num_generations=params.get("num_generations", 4),
                max_prompt_length=params.get("max_prompt_length", 512),
                max_completion_length=params.get("max_completion_length", 256),
                resume_from_checkpoint=params.get("resume_from_checkpoint"),
                load_in_4bit=params.get("load_in_4bit", True),
                extra_callbacks=extra_callbacks,
            )
        elif tool_name == "finetune.train_kto":
            result = await self.finetuner.train_kto_model(
                dataset=dataset_obj,
                output_dir=params["output_dir"],
                base_model=params.get("base_model"),
                num_epochs=params.get("num_epochs", 3),
                beta=params.get("beta", 0.1),
                use_lora=params.get("use_lora", True),
                lora_r=params.get("lora_r", 8),
                desirable_weight=params.get("desirable_weight", 1.0),
                undesirable_weight=params.get("undesirable_weight", 1.0),
                resume_from_checkpoint=params.get("resume_from_checkpoint"),
                load_in_4bit=params.get("load_in_4bit", True),
                extra_callbacks=extra_callbacks,
            )
        else:
            result = await self.finetuner.train_curriculum_model(
                dataset=dataset_obj,
                output_dir=params["output_dir"],
                base_model=params.get("base_model"),
                num_stages=params.get("num_stages", 3),
                num_epochs_per_stage=params.get("num_epochs_per_stage", 1),
                difficulty_order=params.get("difficulty_order", "easy_first"),
                score_column=params.get("score_column", "weighted_score"),
                use_lora=params.get("use_lora", True),
                lora_r=params.get("lora_r", 8),
                lora_alpha=params.get("lora_alpha", 16),
                max_seq_length=params.get("max_seq_length", 2048),
                per_device_train_batch_size=params.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
                gradient_checkpointing=params.get("gradient_checkpointing", False),
                optim=params.get("optim", "adamw_torch"),
                learning_rate=params.get("learning_rate", 2e-4),
                lr_scheduler_type=params.get("lr_scheduler_type", "linear"),
                warmup_ratio=params.get("warmup_ratio", 0.0),
                load_in_4bit=params.get("load_in_4bit", True),
                extra_callbacks=extra_callbacks,
            )

        deploy_result = await self._auto_deploy_if_requested(
            result,
            params.get("deploy", False),
            params.get("deploy_port", 8001),
            params.get("base_model"),
        )
        if deploy_result is not None:
            result["deployment"] = deploy_result
        return result

    # ------------------------------------------------------------------ #
    # Tool registration
    # ------------------------------------------------------------------ #
    def _register_all_tools(self):
        self._register_system_tools()
        self._register_file_tools()
        self._register_extract_tools()
        self._register_generate_tools()
        self._register_clean_tools()
        self._register_normalize_tools()
        self._register_evaluate_tools()
        self._register_model_eval_tools()
        self._register_finetune_tools()
        self._register_training_monitor_tools()
        self._register_test_tools()
        self._register_validate_tools()
        self._register_host_tools()
        self._register_workflow_tools()
        self._register_orchestration_tools()
        self._register_judge_tools()
        self._register_ft_evaluator_tools()
        self._register_dataset_tools()

    # -- System (Resource Checking) --
    def _register_system_tools(self):
        @self.mcp.tool(
            name="system.check_resources",
            description=(
                "Check GPU/RAM/disk status before training. Call this before any "
                "finetune.train* or workflow.* tool to verify hardware is sufficient. "
                "Returns gpu (name, vram_total_gb, vram_free_gb, compute_capability), "
                "ram (total_gb, free_gb, percent_used), and disk (free_gb) info."
            ),
        )
        async def check_resources() -> str:
            result = await self._invoke_dependency(self.finetuner.check_resources)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.preflight_check",
            description=(
                "Estimate VRAM requirements for a training configuration and check if "
                "it will fit on the available GPU. Returns can_run (bool), "
                "estimated_vram_gb, available_vram_gb, headroom_gb, a detailed "
                "breakdown (model_gb, lora_gb, optimizer_gb, activations_gb, "
                "overhead_gb), recommendations for optimal settings, and warnings. "
                "Call this before finetune.train* to avoid OOM errors."
            ),
        )
        async def preflight_check(
            model_name: Optional[str] = None,
            quantization: ResourceQuantization = "4bit",
            batch_size: int = 1,
            max_seq_length: int = 512,
            technique: TechniqueName = "sft",
            use_lora: bool = True,
            lora_r: int = 8,
            gradient_checkpointing: bool = False,
        ) -> str:
            resolved_model = model_name or self.finetuner.config.base_model
            result = await self._invoke_dependency(
                self.finetuner.preflight_check,
                model_name=resolved_model,
                quantization=quantization,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                technique=technique,
                use_lora=use_lora,
                lora_r=lora_r,
                gradient_checkpointing=gradient_checkpointing,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.prescribe",
            description=(
                "Recommend optimal training configuration based on your hardware and dataset. "
                "Analyzes GPU VRAM, RAM, disk space, dataset size, and text length to prescribe "
                "hyperparameters (batch_size, learning_rate, epochs, lora_r, seq_length, etc.) "
                "and dataset sampling if resources are limited. "
                "Call this BEFORE finetune.train* to get a ready-to-use config. "
                "Pass the returned config values directly to finetune.train_async."
            ),
        )
        async def prescribe(
            dataset_path: Optional[str] = None,
            model_name: Optional[str] = None,
            technique: TechniqueName = "sft",
        ) -> str:
            if not dataset_path:
                return json.dumps({
                    "success": False,
                    "error": (
                        "dataset_path is required for system.prescribe. "
                        "Use system.auto_prescribe if you need model suggestions "
                        "before selecting a dataset."
                    ),
                }, indent=2)

            resolved_model = model_name or self.finetuner.config.base_model

            # Get dataset metadata + text length stats
            meta = await self.dataset_service.info(dataset_path)
            if not meta.get("success"):
                return json.dumps(meta, indent=2)

            row_count = meta["metadata"]["row_count"]

            stats = await self.dataset_service.sample_text_stats(dataset_path)
            avg_text_length = stats.get("avg_length", 200) if stats.get("success") else 200

            result = await self._invoke_dependency(
                self.finetuner.prescribe,
                model_name=resolved_model,
                dataset_row_count=row_count,
                dataset_avg_text_length=avg_text_length,
                technique=technique,
            )

            # Enrich with dataset info
            result["dataset_info"] = {
                "file_path": dataset_path,
                "row_count": row_count,
                "columns": meta["metadata"].get("columns", []),
                "technique": meta["metadata"].get("technique"),
                "avg_text_length": avg_text_length,
                "p95_text_length": stats.get("p95_length", 0) if stats.get("success") else 0,
            }
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.setup_check",
            description=(
                "Validate all prerequisites for using MCP Tuna: API keys, GPU, "
                "HuggingFace token, disk space, required Python packages. "
                "Run this first before any pipeline operation."
            ),
        )
        async def setup_check() -> str:
            def _setup_check_sync() -> Dict[str, Any]:
                import os
                import shutil

                checks = []
                def _check(
                    name: str,
                    status: str,
                    detail: str,
                    *,
                    category: str,
                    required: bool = False,
                    action_path: Optional[str] = None,
                    action_label: Optional[str] = None,
                ) -> Dict[str, Any]:
                    return {
                        "name": name,
                        "status": status,
                        "detail": detail,
                        "category": category,
                        "required": required,
                        "action_path": action_path,
                        "action_label": action_label,
                    }

                openai_key = bool(os.getenv("OPENAI_API_KEY"))
                openai_base = (
                    os.getenv("OPENAI_BASE_URL")
                    or os.getenv("OPENAI_API_BASE")
                    or os.getenv("OPENAI_API_BASE_URL")
                )
                anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
                anthropic_base = os.getenv("ANTHROPIC_API_BASE")
                google_key = bool(os.getenv("GOOGLE_API_KEY"))
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
                any_provider = openai_key or anthropic_key or google_key

                checks.extend([
                    _check(
                        "LLM API Provider",
                        "pass" if any_provider else "fail",
                        "At least one provider is configured"
                        if any_provider
                        else "No provider configured. Add OpenAI, Anthropic, or Google credentials.",
                        category="provider",
                        required=True,
                        action_path="/settings#providers",
                        action_label="Configure providers",
                    ),
                    _check(
                        "OpenAI API Key",
                        "pass" if openai_key else "warn",
                        "Configured"
                        if openai_key
                        else "Optional unless you want GPT or OpenAI-compatible models.",
                        category="provider",
                        action_path="/settings#providers",
                        action_label="Edit provider settings",
                    ),
                    _check(
                        "OpenAI Base URL",
                        "pass" if openai_base else "warn",
                        f"Configured: {openai_base}"
                        if openai_base
                        else "Optional. Use for proxies, local gateways, or OpenAI-compatible APIs.",
                        category="provider",
                        action_path="/settings#providers",
                        action_label="Edit provider settings",
                    ),
                    _check(
                        "Anthropic API Key",
                        "pass" if anthropic_key else "warn",
                        "Configured" if anthropic_key else "Optional unless you want Claude models.",
                        category="provider",
                        action_path="/settings#providers",
                        action_label="Edit provider settings",
                    ),
                    _check(
                        "Anthropic Base URL",
                        "pass" if anthropic_base else "warn",
                        f"Configured: {anthropic_base}"
                        if anthropic_base
                        else "Optional. Use for Anthropic-compatible gateways or proxies.",
                        category="provider",
                        action_path="/settings#providers",
                        action_label="Edit provider settings",
                    ),
                    _check(
                        "Google API Key",
                        "pass" if google_key else "warn",
                        "Configured" if google_key else "Optional unless you want Gemini models.",
                        category="provider",
                        action_path="/settings#providers",
                        action_label="Edit provider settings",
                    ),
                    _check(
                        "HF Token",
                        "pass" if hf_token else "warn",
                        "Configured"
                        if hf_token
                        else "Optional, but required for gated models and push_to_hub.",
                        category="provider",
                        action_path="/settings#providers",
                        action_label="Configure providers",
                    ),
                ])

                try:
                    import torch
                    gpu_ok = torch.cuda.is_available()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
                    checks.append(_check(
                        "GPU",
                        "pass" if gpu_ok else "warn",
                        gpu_name or "No GPU detected. Training will be slower and more limited.",
                        category="system",
                        action_path="/settings#environment",
                        action_label="View environment",
                    ))
                except ImportError:
                    checks.append(_check(
                        "GPU",
                        "fail",
                        "torch is not installed, so GPU checks are unavailable.",
                        category="system",
                        required=True,
                        action_path="/settings#environment",
                        action_label="View environment",
                    ))

                disk = shutil.disk_usage(".")
                free_gb = round(disk.free / (1024 ** 3), 1)
                disk_status = "pass" if free_gb > 15 else "warn" if free_gb > 5 else "fail"
                checks.append(_check(
                    "Disk Space",
                    disk_status,
                    f"{free_gb} GB free",
                    category="system",
                    required=True,
                    action_path="/settings#storage",
                    action_label="Review storage",
                ))

                for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
                    try:
                        __import__(pkg)
                        checks.append(_check(
                            f"Package: {pkg}",
                            "pass",
                            "Installed",
                            category="package",
                            required=True,
                            action_path="/settings#environment",
                            action_label="View environment",
                        ))
                    except ImportError:
                        checks.append(_check(
                            f"Package: {pkg}",
                            "fail",
                            "Not installed",
                            category="package",
                            required=True,
                            action_path="/settings#environment",
                            action_label="View environment",
                        ))

                all_passed = all(c["status"] != "fail" for c in checks)
                return {"success": True, "checks": checks, "all_passed": all_passed}

                has_key = bool(os.getenv("OPENAI_API_KEY"))
                checks.append({
                    "name": "OPENAI_API_KEY", "status": "pass" if has_key else "warn",
                    "detail": "Set" if has_key else "Not set — generation/evaluation tools require this",
                })

                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
                checks.append({
                    "name": "HF_TOKEN", "status": "pass" if hf_token else "warn",
                    "detail": "Set" if hf_token else "Not set — required for gated models and push_to_hub",
                })

                try:
                    import torch
                    gpu_ok = torch.cuda.is_available()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
                    checks.append({
                        "name": "GPU", "status": "pass" if gpu_ok else "warn",
                        "detail": gpu_name or "No GPU detected — training will be slow",
                    })
                except ImportError:
                    checks.append({"name": "GPU", "status": "fail", "detail": "torch not installed"})

                disk = shutil.disk_usage(".")
                free_gb = round(disk.free / (1024 ** 3), 1)
                checks.append({
                    "name": "Disk Space", "status": "pass" if free_gb > 10 else "warn",
                    "detail": f"{free_gb} GB free",
                })

                for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
                    try:
                        __import__(pkg)
                        checks.append({"name": f"Package: {pkg}", "status": "pass", "detail": "Installed"})
                    except ImportError:
                        checks.append({"name": f"Package: {pkg}", "status": "fail", "detail": "Not installed"})

                all_passed = all(c["status"] != "fail" for c in checks)
                return {"success": True, "checks": checks, "all_passed": all_passed}

            result = await self._invoke_dependency(_setup_check_sync)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.config",
            description="Show current gateway configuration (model defaults, output dirs, env vars).",
        )
        async def show_config() -> str:
            import os
            return json.dumps({
                "success": True,
                "config": self._config,
                "env": {
                    "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else None,
                    "OPENAI_API_BASE": (
                        os.getenv("OPENAI_BASE_URL")
                        or os.getenv("OPENAI_API_BASE")
                        or os.getenv("OPENAI_API_BASE_URL")
                    ),
                    "ANTHROPIC_API_KEY": "***" if os.getenv("ANTHROPIC_API_KEY") else None,
                    "ANTHROPIC_API_BASE": os.getenv("ANTHROPIC_API_BASE"),
                    "GOOGLE_API_KEY": "***" if os.getenv("GOOGLE_API_KEY") else None,
                    "HF_TOKEN": "***" if os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") else None,
                },
            }, indent=2)
            return json.dumps({
                "success": True,
                "config": self._config,
                "env": {
                    "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else None,
                    "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
                    "HF_TOKEN": "***" if os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") else None,
                },
            }, indent=2)

        @self.mcp.tool(
            name="system.set_hf_token",
            description=(
                "Set the HuggingFace Hub token at runtime for gated model access. "
                "Validates the token and returns the associated username. "
                "Note: token is in-memory only and resets on gateway restart."
            ),
        )
        async def set_hf_token(token: str) -> str:
            def _set_hf_token_sync() -> Dict[str, Any]:
                import os as _os

                _os.environ["HF_TOKEN"] = token
                _os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=token)
                    user = api.whoami()
                    return {
                        "success": True,
                        "username": user.get("name", "unknown"),
                        "message": "HF token set and validated successfully",
                    }
                except ImportError:
                    return {
                        "success": True,
                        "message": "HF token set (could not validate — huggingface_hub not installed)",
                    }
                except Exception as e:
                    return {
                        "success": True,
                        "warning": f"Token set but validation failed: {e!s}",
                    }

            result = await self._invoke_dependency(_set_hf_token_sync)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.set_runtime_env",
            description=(
                "Set or clear supported provider environment variables at runtime. "
                "Useful for API keys and provider base URLs in the dashboard settings. "
                "Changes are in-memory only and reset on gateway restart."
            ),
        )
        async def set_runtime_env(key: str, value: Optional[str] = None) -> str:
            def _set_runtime_env_sync() -> Dict[str, Any]:
                import os as _os

                aliases = {
                    "OPENAI_API_KEY": ["OPENAI_API_KEY"],
                    "OPENAI_API_BASE": ["OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_BASE_URL"],
                    "ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY"],
                    "ANTHROPIC_API_BASE": ["ANTHROPIC_API_BASE"],
                    "GOOGLE_API_KEY": ["GOOGLE_API_KEY"],
                    "HF_TOKEN": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
                }
                if key not in aliases:
                    return {"success": False, "error": f"Unsupported runtime env key: {key}"}

                normalized = value.strip() if isinstance(value, str) else ""
                for target in aliases[key]:
                    if normalized:
                        _os.environ[target] = normalized
                    else:
                        _os.environ.pop(target, None)

                return {
                    "success": True,
                    "key": key,
                    "configured": bool(normalized),
                    "message": (
                        f"{key} updated for the current gateway process."
                        if normalized
                        else f"{key} cleared from the current gateway process."
                    ),
                }

            result = await self._invoke_dependency(_set_runtime_env_sync)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.health",
            description=(
                "Unified health dashboard: GPU/RAM/disk status, active training jobs, "
                "active deployments, and an overall health score (green/yellow/red). "
                "Call this to get a quick overview of system state."
            ),
        )
        async def system_health() -> str:
            resources = await self._invoke_dependency(self.finetuner.check_resources)

            # Active training jobs
            job_manager = self._job_manager_instance
            jobs = await job_manager.alist_jobs(limit=100) if job_manager else []
            running_jobs = [j for j in jobs if getattr(j, "status", None) == "running"]

            # Active deployments
            deployments_info: dict = {"deployments": [], "count": 0}
            if self._hosting_svc is not None:
                deployments_info = await self._hosting_svc.list_deployments()

            # Health scoring
            warnings: list = []
            gpu = resources.get("gpu", {})
            gpus = resources.get("gpus", [])
            ram = resources.get("ram", {})
            disk = resources.get("disk", {})

            status = "green"
            gpu_devices = gpus or ([gpu] if gpu.get("available") else [])
            if gpu_devices:
                hottest_gpu = None
                hottest_vram_pct = 0.0
                for device in gpu_devices:
                    total = device.get("vram_total_gb", 0) or 0
                    if total <= 0:
                        continue
                    used = max(
                        float(device.get("vram_used_gb", 0) or 0),
                        float(device.get("vram_reserved_gb", 0) or 0),
                    )
                    used_pct = (used / total) * 100
                    if used_pct >= hottest_vram_pct:
                        hottest_vram_pct = used_pct
                        hottest_gpu = device

                if hottest_vram_pct > 80:
                    status = "red"
                    warnings.append(
                        f"GPU {hottest_gpu.get('index', 0)} VRAM {hottest_vram_pct:.0f}% used"
                    )
                elif hottest_vram_pct > 60:
                    status = "yellow"
                    warnings.append(
                        f"GPU {hottest_gpu.get('index', 0)} VRAM {hottest_vram_pct:.0f}% used"
                    )

            ram_pct = ram.get("percent_used", 0)
            if ram_pct > 90:
                status = "red"
                warnings.append(f"RAM {ram_pct:.0f}% used")
            elif ram_pct > 75:
                if status != "red":
                    status = "yellow"
                warnings.append(f"RAM {ram_pct:.0f}% used")

            disk_free = disk.get("free_gb", 100)
            if disk_free < 5:
                status = "red"
                warnings.append(f"Disk critically low: {disk_free:.1f} GB free")
            elif disk_free < 15:
                if status != "red":
                    status = "yellow"
                warnings.append(f"Disk low: {disk_free:.1f} GB free")

            return json.dumps({
                "success": True,
                "status": status,
                "resources": resources,
                "active_training_jobs": len(running_jobs),
                "active_deployments": deployments_info.get("count", 0),
                "warnings": warnings,
            }, indent=2)

        @self.mcp.tool(
            name="system.clear_gpu_cache",
            description=(
                "Free the current MCP process's PyTorch GPU cache without stopping "
                "deployments or deleting data. Use this after training/inference to "
                "recover VRAM held by this process."
            ),
        )
        async def clear_gpu_cache() -> str:
            result = await self._invoke_dependency(self.finetuner.clear_gpu_memory)
            if result.get("success"):
                result["message"] = (
                    "Cleared the current process GPU cache. VRAM used by other processes "
                    "or active deployments may remain allocated."
                )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.clear_all",
            description=(
                "Emergency recovery: stop ALL deployments and clear GPU memory. "
                "Use this when the system is in a bad state (OOM, stuck deployments). "
                "Returns a summary of what was cleaned up."
            ),
        )
        async def clear_all() -> str:
            results: dict = {"deployments_stopped": 0, "gpu_cleared": False}

            # Stop all deployments
            if self._hosting_svc is not None:
                dep_ids = list(self._hosting_svc._deployments.keys())
                for dep_id in dep_ids:
                    try:
                        await self._hosting_svc.stop_deployment(dep_id)
                    except Exception:
                        pass
                results["deployments_stopped"] = len(dep_ids)

            # Clear GPU
            try:
                gpu_result = await self._invoke_dependency(self.finetuner.clear_gpu_memory)
                results["gpu_cleared"] = gpu_result.get("success", False)
                results["gpu_stats"] = gpu_result.get("memory_stats", {})
            except Exception as e:
                results["gpu_cleared"] = False
                results["gpu_error"] = str(e)

            results["success"] = True
            return json.dumps(results, indent=2)

        @self.mcp.tool(
            name="system.disk_preflight",
            description=(
                "Check if there is enough free disk space for an operation. "
                "Pass estimated_size_gb to validate before training (checkpoints) "
                "or model downloads."
            ),
        )
        async def disk_preflight(
            output_dir: str = "",
            estimated_size_gb: float = 5.0,
        ) -> str:
            result = await self._invoke_dependency(
                self.finetuner.disk_preflight,
                output_dir=output_dir,
                estimated_size_gb=estimated_size_gb,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.prescribe_pipeline",
            description=(
                "End-to-end resource recommendations across the full pipeline. "
                "Analyzes feasibility for evaluate, train, and deploy stages based on "
                "your hardware (GPU, RAM, disk) and dataset. Returns per-stage "
                "feasibility, recommended configs, and warnings. "
                "Call this BEFORE starting a full pipeline run."
            ),
        )
        async def prescribe_pipeline(
            dataset_path: str,
            model_name: Optional[str] = None,
            technique: TechniqueName = "sft",
            stages: Optional[str] = None,
        ) -> str:
            resolved_model = model_name or self.finetuner.config.base_model

            meta = await self.dataset_service.info(dataset_path)
            if not meta.get("success"):
                return json.dumps(meta, indent=2)

            row_count = meta["metadata"]["row_count"]
            stats = await self.dataset_service.sample_text_stats(dataset_path)
            avg_text_length = stats.get("avg_length", 200) if stats.get("success") else 200

            stage_list = stages.split(",") if stages else None

            result = await self._invoke_dependency(
                self.finetuner.prescribe_pipeline,
                model_name=resolved_model,
                dataset_row_count=row_count,
                dataset_avg_text_length=avg_text_length,
                stages=stage_list,
                technique=technique,
            )

            result["dataset_info"] = {
                "file_path": dataset_path,
                "row_count": row_count,
                "avg_text_length": avg_text_length,
            }
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.auto_prescribe",
            description=(
                "Automatically recommend the best model AND training configuration "
                "for your hardware. Unlike system.prescribe (which requires a model_name), "
                "this tool detects GPU VRAM, filters known models by what fits, ranks by "
                "use_case, and generates ready-to-use training configs for top candidates. "
                "Pass either dataset_path OR (dataset_row_count + dataset_avg_text_length)."
            ),
        )
        async def auto_prescribe(
            dataset_row_count: int = 0,
            dataset_avg_text_length: int = 0,
            dataset_path: Optional[str] = None,
            technique: TechniqueName = "sft",
            use_case: ModelUseCase = "general",
        ) -> str:
            import os
            from pathlib import Path as _Path

            technique_warning = None
            if dataset_path:
                dp = _Path(dataset_path)
                # If a directory, find a dataset file — prefer one matching technique
                if dp.is_dir():
                    supported = (".jsonl", ".json", ".csv", ".parquet")
                    candidates = [
                        f for f in sorted(dp.iterdir())
                        if f.is_file() and f.suffix.lower() in supported
                    ]
                    if not candidates:
                        return json.dumps({
                            "success": False,
                            "error": (
                                f"No dataset files found in {dataset_path}. "
                                f"Supported formats: {', '.join(supported)}"
                            ),
                        }, indent=2)
                    best = candidates[0]
                    for c in candidates:
                        c_meta = await self.dataset_service.info(str(c))
                        if (
                            c_meta.get("success")
                            and c_meta["metadata"].get("technique") == technique
                        ):
                            best = c
                            break
                    dataset_path = str(best)

                meta = await self.dataset_service.info(dataset_path)
                if not meta.get("success"):
                    return json.dumps(meta, indent=2)

                detected = meta["metadata"].get("technique")
                technique_warning = None
                if detected and detected != technique:
                    technique_warning = (
                        f"Dataset appears to be '{detected}' format but "
                        f"'{technique}' was requested. "
                        f"Columns: {meta['metadata'].get('columns', [])}"
                    )

                dataset_row_count = meta["metadata"]["row_count"]
                stats = await self.dataset_service.sample_text_stats(dataset_path)
                dataset_avg_text_length = (
                    stats.get("avg_length", 200) if stats.get("success") else 200
                )

            if dataset_row_count <= 0:
                return json.dumps({
                    "success": False,
                    "error": (
                        "Provide dataset_path or dataset_row_count + "
                        "dataset_avg_text_length"
                    ),
                }, indent=2)

            result = await self._invoke_dependency(
                self.finetuner.auto_prescribe,
                dataset_row_count=dataset_row_count,
                dataset_avg_text_length=dataset_avg_text_length,
                technique=technique,
                use_case=use_case,
            )
            if dataset_path:
                result["resolved_dataset"] = dataset_path
            if dataset_path and technique_warning:
                result["technique_warning"] = technique_warning
            return json.dumps(result, indent=2)

    def _register_file_tools(self):
        @self.mcp.tool(
            name="file.list_deployment_roots",
            description=(
                "List safe server-side roots for browsing model and adapter folders. "
                "Includes workspace, output, uploads, and Hugging Face cache when available."
            ),
        )
        async def file_list_deployment_roots() -> str:
            return json.dumps({
                "success": True,
                "roots": self._deployment_browser_roots(),
            }, indent=2)

        @self.mcp.tool(
            name="file.browse_deployment_dir",
            description=(
                "Browse directories within a safe server-side root for deployment path selection. "
                "Returns the current absolute folder path and child entries."
            ),
        )
        async def file_browse_deployment_dir(root_id: str, path: str = ".") -> str:
            return json.dumps(self._browse_deployment_directory(root_id, path), indent=2)

        @self.mcp.tool(
            name="file.upload",
            description=(
                "Upload a browser-selected file into the server uploads directory. "
                "Returns both relative path and absolute file_path for downstream "
                "tools like extract.load_document and generate.from_document."
            ),
        )
        async def file_upload(filename: str, content_base64: str) -> str:
            relative_path = str(Path(filename))
            result = await self.file_service.upload(relative_path, content_base64)
            return json.dumps(result, indent=2)

    # -- Extract --
    def _register_extract_tools(self):
        @self.mcp.tool(name="extract.load_document",
                       description="Load and parse a document file (PDF, Markdown, etc.)")
        async def load_document(file_path: str) -> str:
            # Avoid initializing the generator LLM provider for pure document loading.
            from data_generator_pipeline.loaders import get_loader

            try:
                loader = get_loader(file_path)
                file_name, pages = await self._invoke_dependency(loader.load, file_path)
                result = {
                    "success": True,
                    "file_name": file_name,
                    "file_path": file_path,
                    "pages": pages,
                    "total_pages": len(pages),
                }
            except Exception as e:
                result = {"success": False, "error": str(e), "file_path": file_path}

            return json.dumps(result, indent=2)

    # -- Generate --
    def _register_generate_tools(self):
        @self.mcp.tool(name="generate.from_document",
                       description="Generate fine-tuning data from an entire document")
        async def gen_from_doc(
            technique: TechniqueName, file_path: str,
            custom_template: Optional[str] = None,
            start_page: Optional[int] = None, end_page: Optional[int] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_document(
                technique=technique, file_path=file_path,
                custom_template=custom_template,
                start_page=start_page, end_page=end_page,
            ), indent=2)

        @self.mcp.tool(name="generate.from_page",
                       description="Generate fine-tuning data from a single page")
        async def gen_from_page(
            technique: TechniqueName, page_text: str, page_index: int, file_name: str,
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_page(
                technique=technique, page_text=page_text,
                page_index=page_index, file_name=file_name,
                custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(name="generate.batch",
                       description="Generate fine-tuning data from multiple documents")
        async def gen_batch(
            technique: TechniqueName, file_paths: List[str],
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_batch(
                technique=technique, file_paths=file_paths,
                custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(name="generate.list_techniques",
                       description="List available fine-tuning techniques")
        async def list_techniques() -> str:
            from data_generator_pipeline.generators.registry import GENERATOR_REGISTRY
            return json.dumps({
                "success": True,
                "techniques": list(GENERATOR_REGISTRY.keys()),
            }, indent=2)

        @self.mcp.tool(name="generate.get_schema",
                       description="Get the data schema for a fine-tuning technique")
        async def get_schema(technique: TechniqueName) -> str:
            from data_generator_pipeline.generators.registry import GENERATOR_REGISTRY
            import dataclasses
            if technique not in GENERATOR_REGISTRY:
                return json.dumps({"success": False, "error": f"Unknown technique: {technique}"}, indent=2)
            _, datapoint_class = GENERATOR_REGISTRY[technique]
            schema = {
                field.name: {
                    "type": str(field.type),
                    "required": field.default == dataclasses.MISSING,
                }
                for field in dataclasses.fields(datapoint_class)
            }
            return json.dumps({"success": True, "technique": technique, "schema": schema}, indent=2)

        @self.mcp.tool(name="generate.get_template",
                       description="Get the default prompt template for a fine-tuning technique")
        async def get_template(technique: TechniqueName) -> str:
            return json.dumps(self.generator.get_template(technique), indent=2)

        @self.mcp.tool(
            name="generate.from_text",
            description=(
                "Generate fine-tuning data from raw text (no file required). "
                "Useful when text is already in memory or pasted by the user. "
                "Supports all techniques: sft, dpo, grpo, kto."
            ),
        )
        async def gen_from_text(
            technique: TechniqueName,
            text: str,
            source_name: str = "raw_text",
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_page(
                technique=technique, page_text=text,
                page_index=0, file_name=source_name,
                custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(
            name="generate.from_hf_dataset",
            description=(
                "Load a dataset from the HuggingFace Hub and return as MCP Tuna "
                "data_points. Automatically maps columns if the dataset uses "
                "standard naming (instruction/input/output or prompt/chosen/rejected). "
                "Use column_mapping JSON to override: e.g., "
                '\'{"question": "instruction", "answer": "output"}\'. '
                "Returns data_points ready for clean.dataset, evaluate.dataset, "
                "or dataset.save."
            ),
        )
        async def gen_from_hf_dataset(
            dataset_name: str,
            split: str = "train",
            subset: Optional[str] = None,
            max_rows: Optional[int] = None,
            column_mapping: Optional[str] = None,
        ) -> str:
            try:
                from datasets import load_dataset as hf_load_dataset
                ds = hf_load_dataset(dataset_name, subset, split=split)
                if max_rows:
                    ds = ds.select(range(min(max_rows, len(ds))))

                mapping = json.loads(column_mapping) if column_mapping else {}
                data_points = []
                for row in ds:
                    dp = {}
                    for src_col, val in row.items():
                        target_col = mapping.get(src_col, src_col)
                        if isinstance(val, (str, int, float, bool, list)):
                            dp[target_col] = val
                        else:
                            dp[target_col] = str(val)
                    data_points.append(dp)

                return json.dumps({
                    "success": True,
                    "dataset_name": dataset_name,
                    "split": split,
                    "data_points": data_points,
                    "count": len(data_points),
                    "original_columns": list(ds.column_names),
                }, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, indent=2)

    # -- Clean --
    def _register_clean_tools(self):
        @self.mcp.tool(name="clean.dataset",
                       description="Run all cleaning steps on a dataset")
        async def clean_dataset(
            data_points: List[Dict],
            remove_duplicates: bool = True,
            min_instruction_length: int = 10,
            min_output_length: int = 20,
        ) -> str:
            config = CleaningConfig(
                remove_duplicates=remove_duplicates,
                min_instruction_length=min_instruction_length,
                min_output_length=min_output_length,
            )
            return json.dumps(await self.cleaner.clean_dataset(data_points, config), indent=2)

        @self.mcp.tool(name="clean.deduplicate",
                       description="Remove duplicate entries by key")
        async def deduplicate(data_points: List[Dict], key: str = "instruction") -> str:
            return json.dumps(await self.cleaner.deduplicate(data_points, key), indent=2)

        @self.mcp.tool(name="clean.validate_schema",
                       description="Validate entries have required fields for a technique")
        async def validate_schema(data_points: List[Dict], technique: TechniqueName = "sft") -> str:
            return json.dumps(await self.cleaner.validate_schema(data_points, technique), indent=2)

        @self.mcp.tool(name="clean.remove_short",
                       description="Filter entries below length thresholds")
        async def remove_short(
            data_points: List[Dict], min_instruction: int = 10, min_output: int = 20,
        ) -> str:
            return json.dumps(await self.cleaner.remove_short_entries(
                data_points, min_instruction, min_output,
            ), indent=2)

    # -- Normalize --
    def _register_normalize_tools(self):
        @self.mcp.tool(name="normalize.dataset",
                       description="Apply all normalization steps to a dataset")
        async def normalize_dataset(
            data_points: List[Dict],
            target_format: TechniqueName = "sft",
            merge_instruction_input: bool = True,
        ) -> str:
            config = NormalizationConfig(
                target_format=target_format,
                merge_instruction_input=merge_instruction_input,
            )
            return json.dumps(await self.normalizer.normalize_dataset(data_points, config), indent=2)

        @self.mcp.tool(name="normalize.merge_fields",
                       description="Merge instruction + input into a single field")
        async def merge_fields(data_points: List[Dict]) -> str:
            return json.dumps(await self.normalizer.merge_instruction_input(data_points), indent=2)

        @self.mcp.tool(name="normalize.standardize_keys",
                       description="Rename keys to match target format")
        async def standardize_keys(data_points: List[Dict], target_format: TechniqueName = "sft") -> str:
            return json.dumps(await self.normalizer.standardize_keys(data_points, target_format), indent=2)

        @self.mcp.tool(name="normalize.strip_text",
                       description="Strip whitespace and normalize unicode")
        async def strip_text(data_points: List[Dict]) -> str:
            return json.dumps(await self.normalizer.strip_and_clean_text(data_points), indent=2)

    # -- Evaluate --
    def _register_evaluate_tools(self):
        @self.mcp.tool(name="evaluate.dataset",
                       description="Score dataset with complexity, IFD, and quality metrics")
        async def evaluate_dataset(
            data_points: List[Dict],
            metrics: Optional[List[str]] = None,
        ) -> str:
            return json.dumps(
                await self.evaluator.evaluate_dataset(data_points, metrics=metrics),
                indent=2,
            )

        @self.mcp.tool(name="evaluate.filter_by_quality",
                       description="Return entries above quality threshold")
        async def filter_quality(
            data_points: List[Dict],
            threshold: float = 0.7,
            metrics: Optional[List[str]] = None,
        ) -> str:
            return json.dumps(
                await self.evaluator.filter_by_quality(
                    data_points, threshold, metrics=metrics,
                ),
                indent=2,
            )

        @self.mcp.tool(name="evaluate.statistics",
                       description="Return per-metric statistics (min/max/mean/stdev)")
        async def statistics(
            data_points: List[Dict],
            metrics: Optional[List[str]] = None,
        ) -> str:
            return json.dumps(
                await self.evaluator.analyze_statistics(data_points, metrics=metrics),
                indent=2,
            )

        @self.mcp.tool(name="evaluate.list_metrics",
                       description="List all registered evaluation metrics")
        async def list_metrics() -> str:
            result = await self._invoke_dependency(self.evaluator.list_metrics)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate.get_config",
            description="Show the current evaluator config (weights, threshold, language)",
        )
        async def get_evaluate_config() -> str:
            result = await self._invoke_dependency(self.evaluator.get_config)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate.update_config",
            description="Update evaluator config for this gateway session",
        )
        async def update_evaluate_config(
            weights: Optional[Dict[str, float]] = None,
            threshold: Optional[float] = None,
            language: Optional[str] = None,
        ) -> str:
            return json.dumps(
                await self.evaluator.update_config(
                    weights=weights,
                    threshold=threshold,
                    language=language,
                ),
                indent=2,
            )

    # -- Model Evaluate --
    def _register_model_eval_tools(self):
        @self.mcp.tool(
            name="evaluate_model.single",
            description="Score a single generated output against a reference with ROUGE, BERTScore, and LLM-as-Judge",
        )
        async def eval_model_single(
            question: str,
            generated: str,
            reference: str,
            metrics: Optional[List[str]] = None,
        ) -> str:
            result = await self.model_evaluator.evaluate_single(
                question=question,
                generated=generated,
                reference=reference,
                metrics=metrics,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate_model.batch",
            description="Run evaluation on a test set -- optionally runs inference first if model_path is provided",
        )
        async def eval_model_batch(
            test_data: List[Dict],
            metrics: Optional[List[str]] = None,
            model_path: Optional[str] = None,
            adapter_path: Optional[str] = None,
            max_new_tokens: int = 1024,
            flatten: bool = False,
        ) -> str:
            result = await self.model_evaluator.evaluate_batch(
                test_data=test_data,
                metrics=metrics,
                model_path=model_path,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                flatten=flatten,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate_model.export",
            description="Export evaluation results as JSONL, JSON, or Excel",
        )
        async def eval_model_export(
            results: List[Dict],
            output_path: str,
            format: ModelEvalExportFormat = "jsonl",
        ) -> str:
            result = await self.model_evaluator.export_results(
                results=results,
                output_path=output_path,
                format=format,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate_model.summary",
            description="Compute aggregate statistics (min/max/mean/stdev) for evaluation results",
        )
        async def eval_model_summary(results: List[Dict]) -> str:
            summary = await self._invoke_dependency(
                self.model_evaluator.compute_summary, results,
            )
            return json.dumps({"success": True, "summary": summary}, indent=2)

    # -- Finetune --
    def _register_finetune_tools(self):
        @self.mcp.tool(
            name="finetune.train",
            description=(
                "Fine-tune a model using a dataset file (SFT with QLoRA). "
                "Returns model_path usable as base_model in finetune.train_dpo, "
                "finetune.train_grpo, finetune.train_kto, or finetune.train_curriculum. "
                "Set deploy=True to auto-deploy after training. "
                "After deployment, use host.chat to let the user chat with the model, "
                "or share the chat_command from the deployment result for CLI access."
            ),
        )
        async def train(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            completion_only_loss: bool = True,
            early_stopping_patience: Optional[int] = None,
            eval_file_path: Optional[str] = None,
            push_to_hub: Optional[str] = None,
            lr_scheduler_type: str = "linear",
            warmup_ratio: float = 0.0,
            weight_decay: float = 0.0,
            max_grad_norm: float = 1.0,
            learning_rate: float = 2e-4,
            report_to: Optional[str] = None,
            max_seq_length: int = 2048,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            gradient_checkpointing: bool = False,
            optim: str = "adamw_torch",
            load_in_4bit: bool = True,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                completion_only_loss=completion_only_loss,
                early_stopping_patience=early_stopping_patience,
                eval_file_path=eval_file_path,
                push_to_hub=push_to_hub,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                learning_rate=learning_rate,
                report_to=report_to or [],
                max_seq_length=max_seq_length,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                load_in_4bit=load_in_4bit,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_vlm",
            description=(
                "Fine-tune a vision-language model using a multimodal dataset manifest. "
                "Dataset rows must include canonical 'messages' with image blocks and assistant text. "
                "This path is additive and does not replace the existing text SFT trainer."
            ),
        )
        async def train_vlm(
            dataset_path: str,
            output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            learning_rate: float = 2e-4,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            max_seq_length: int = 2048,
            load_in_4bit: bool = True,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_vlm_model(
                dataset=load_result["dataset_object"],
                dataset_path=dataset_path,
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_dpo",
            description=(
                "Fine-tune a model with DPO (Direct Preference Optimization) -- "
                "dataset needs prompt/chosen/rejected columns. "
                "Accepts base_model from a previous finetune.train result (model_path). "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_dpo(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            load_in_4bit: bool = True,
            resume_from_checkpoint: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_dpo_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                beta=beta,
                use_lora=use_lora,
                lora_r=lora_r,
                load_in_4bit=load_in_4bit,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_grpo",
            description=(
                "Fine-tune a model with GRPO (Group Relative Policy Optimization) -- "
                "dataset needs prompt/responses/rewards columns. "
                "Accepts base_model from a previous finetune.train result (model_path). "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_grpo(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            num_generations: int = 4,
            load_in_4bit: bool = True,
            resume_from_checkpoint: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_grpo_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                num_generations=num_generations,
                load_in_4bit=load_in_4bit,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_kto",
            description=(
                "Fine-tune a model with KTO (Kahneman-Tversky Optimization) -- "
                "dataset needs prompt/completion/label columns. "
                "Accepts base_model from a previous finetune.train result (model_path). "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_kto(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            load_in_4bit: bool = True,
            resume_from_checkpoint: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_kto_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                beta=beta,
                use_lora=use_lora,
                lora_r=lora_r,
                load_in_4bit=load_in_4bit,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_curriculum",
            description=(
                "Curriculum fine-tune: auto-scores dataset by difficulty, trains easy-to-hard "
                "in stages. Each stage's output model feeds into the next stage automatically. "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_curriculum(
            dataset_path: str,
            output_dir: str,
            base_model: Optional[str] = None,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            difficulty_order: DifficultyOrder = "easy_first",
            score_column: str = "weighted_score",
            use_lora: bool = True,
            lora_r: int = 8,
            max_seq_length: int = 2048,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            gradient_checkpointing: bool = False,
            optim: str = "adamw_torch",
            learning_rate: float = 2e-4,
            lr_scheduler_type: str = "linear",
            warmup_ratio: float = 0.0,
            load_in_4bit: bool = True,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_curriculum_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_stages=num_stages,
                num_epochs_per_stage=num_epochs_per_stage,
                difficulty_order=difficulty_order,
                score_column=score_column,
                use_lora=use_lora,
                lora_r=lora_r,
                max_seq_length=max_seq_length,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                learning_rate=learning_rate,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                load_in_4bit=load_in_4bit,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.sequential_train",
            description=(
                "Chain multiple training techniques sequentially (e.g., SFT -> DPO -> GRPO). "
                "Each stage's output model_path automatically becomes the next stage's base_model. "
                "Accepts a JSON list of stages, each with technique, dataset_path, and optional "
                "hyperparameters. Returns per-stage results and final_model_path for deployment "
                "via host.deploy_mcp or evaluation via evaluate_model.batch."
            ),
        )
        async def sequential_train(
            stages: str,
            output_dir: str,
            base_model: Optional[str] = None,
            merge_between_stages: bool = True,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            parsed_stages = json.loads(stages) if isinstance(stages, str) else stages
            result = await self.finetuner.train_sequential(
                stages=parsed_stages,
                output_dir=output_dir,
                base_model=base_model,
                merge_between_stages=merge_between_stages,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.load_dataset",
                       description="Load a dataset from file for fine-tuning")
        async def load_dataset(file_path: str, format: DatasetFormat = "jsonl") -> str:
            result = await self.finetuner.load_dataset_from_file(file_path, format)
            if "dataset_object" in result:
                del result["dataset_object"]
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.prepare_dataset",
                       description="Prepare inline data for fine-tuning")
        async def prepare_dataset(data: str) -> str:
            data_list = json.loads(data) if isinstance(data, str) else data
            result = await self.finetuner.prepare_dataset(data_list)
            if "dataset_object" in result:
                del result["dataset_object"]
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.merge_adapter",
            description=(
                "Merge a LoRA adapter into the base model to produce a standalone "
                "model directory. Optionally push the merged model to HuggingFace Hub."
            ),
        )
        async def merge_adapter(
            base_model: str,
            adapter_path: str,
            output_path: str,
            push_to_hub: Optional[str] = None,
        ) -> str:
            return json.dumps(
                await self.finetuner.merge_adapter(
                    base_model=base_model, adapter_path=adapter_path,
                    output_path=output_path, push_to_hub=push_to_hub,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="finetune.export_gguf",
            description=(
                "Export a model to GGUF format for use with llama.cpp or Ollama. "
                "Requires the llama-cpp-python package (pip install mcp-tuna[export]). "
                "Supported quantizations: q4_0, q4_k_m, q5_k_m, q8_0, f16."
            ),
        )
        async def export_gguf(
            model_path: str,
            output_path: str,
            quantization: GGUFQuantization = "q4_k_m",
        ) -> str:
            return json.dumps(
                await self.finetuner.export_gguf(
                    model_path=model_path, output_path=output_path,
                    quantization=quantization,
                ),
                indent=2,
            )

    # -- Training Monitor --
    def _register_training_monitor_tools(self):
        @self.mcp.tool(
            name="finetune.train_async",
            description=(
                "Start SFT fine-tuning in the background and return a job_id immediately. "
                "Use finetune.job_status(job_id) to poll progress (step, loss, ETA, GPU). "
                "Use finetune.cancel_job(job_id) to stop training early. "
                "Same parameters as finetune.train but non-blocking."
            ),
        )
        async def train_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            completion_only_loss: bool = True,
            early_stopping_patience: Optional[int] = None,
            eval_file_path: Optional[str] = None,
            push_to_hub: Optional[str] = None,
            lr_scheduler_type: str = "linear",
            warmup_ratio: float = 0.0,
            weight_decay: float = 0.0,
            max_grad_norm: float = 1.0,
            learning_rate: float = 2e-4,
            report_to: Optional[str] = None,
            max_seq_length: int = 2048,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            gradient_checkpointing: bool = False,
            optim: str = "adamw_torch",
            load_in_4bit: bool = True,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="sft",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_epochs": num_epochs, "lora_r": lora_r,
                    "batch_size": per_device_train_batch_size,
                    "learning_rate": learning_rate,
                    "lora_dropout": lora_dropout,
                    "load_in_4bit": load_in_4bit,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    completion_only_loss=completion_only_loss,
                    early_stopping_patience=early_stopping_patience,
                    eval_file_path=eval_file_path,
                    push_to_hub=push_to_hub,
                    lr_scheduler_type=lr_scheduler_type,
                    warmup_ratio=warmup_ratio,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    learning_rate=learning_rate,
                    report_to=report_to or [],
                    max_seq_length=max_seq_length,
                    per_device_train_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    gradient_checkpointing=gradient_checkpointing,
                    optim=optim,
                    load_in_4bit=load_in_4bit,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True,
                "job_id": job.job_id,
                "status": "running",
                "message": "Training started. Use finetune.job_status to monitor progress.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_vlm_async",
            description=(
                "Start VLM supervised fine-tuning in the background and return a job_id immediately. "
                "Dataset rows must use the canonical multimodal 'messages' schema."
            ),
        )
        async def train_vlm_async(
            dataset_path: str,
            output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            learning_rate: float = 2e-4,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            max_seq_length: int = 2048,
            load_in_4bit: bool = True,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="vlm_sft",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_epochs": num_epochs,
                    "lora_r": lora_r,
                    "batch_size": per_device_train_batch_size,
                    "learning_rate": learning_rate,
                    "load_in_4bit": load_in_4bit,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_vlm_model(
                    dataset=dataset_obj,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    learning_rate=learning_rate,
                    per_device_train_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True,
                "job_id": job.job_id,
                "status": "running",
                "message": "VLM training started. Use finetune.job_status to monitor progress.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_dpo_async",
            description=(
                "Start DPO fine-tuning in the background. Returns job_id. "
                "Dataset needs prompt/chosen/rejected columns. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_dpo_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            load_in_4bit: bool = True,
            resume_from_checkpoint: Optional[str] = None,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="dpo",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_epochs": num_epochs,
                    "beta": beta,
                    "load_in_4bit": load_in_4bit,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_dpo_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    beta=beta,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    load_in_4bit=load_in_4bit,
                    resume_from_checkpoint=resume_from_checkpoint,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "DPO training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_grpo_async",
            description=(
                "Start GRPO fine-tuning in the background. Returns job_id. "
                "Dataset needs prompt/responses/rewards columns. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_grpo_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            num_generations: int = 4,
            max_prompt_length: int = 512,
            max_completion_length: int = 256,
            load_in_4bit: bool = True,
            resume_from_checkpoint: Optional[str] = None,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="grpo",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_epochs": num_epochs,
                    "num_generations": num_generations,
                    "load_in_4bit": load_in_4bit,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_grpo_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    num_generations=num_generations,
                    max_prompt_length=max_prompt_length,
                    max_completion_length=max_completion_length,
                    load_in_4bit=load_in_4bit,
                    resume_from_checkpoint=resume_from_checkpoint,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "GRPO training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_kto_async",
            description=(
                "Start KTO fine-tuning in the background. Returns job_id. "
                "Dataset needs prompt/completion/label columns. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_kto_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            desirable_weight: float = 1.0,
            undesirable_weight: float = 1.0,
            load_in_4bit: bool = True,
            resume_from_checkpoint: Optional[str] = None,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="kto",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_epochs": num_epochs,
                    "beta": beta,
                    "load_in_4bit": load_in_4bit,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_kto_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    beta=beta,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    desirable_weight=desirable_weight,
                    undesirable_weight=undesirable_weight,
                    load_in_4bit=load_in_4bit,
                    resume_from_checkpoint=resume_from_checkpoint,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "KTO training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_curriculum_async",
            description=(
                "Start curriculum learning in the background. Returns job_id. "
                "Scores dataset, buckets by difficulty, trains stage-by-stage. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_curriculum_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            score_column: str = "weighted_score",
            difficulty_order: DifficultyOrder = "easy_first",
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            load_in_4bit: bool = True,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="curriculum",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_stages": num_stages,
                    "epochs_per_stage": num_epochs_per_stage,
                    "load_in_4bit": load_in_4bit,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_curriculum_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_stages=num_stages,
                    num_epochs_per_stage=num_epochs_per_stage,
                    score_column=score_column,
                    difficulty_order=difficulty_order,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    load_in_4bit=load_in_4bit,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "Curriculum training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.sequential_train_async",
            description=(
                "Start sequential multi-technique training in the background. Returns job_id. "
                "Chains SFT -> DPO -> GRPO -> KTO stages, auto-merging LoRA between them. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def sequential_train_async(
            stages: str, output_dir: str,
            base_model: Optional[str] = None,
            merge_between_stages: bool = True,
        ) -> str:
            try:
                stages_list = json.loads(stages)
            except json.JSONDecodeError as e:
                return json.dumps({"success": False, "error": f"Invalid stages JSON: {e}"}, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            techniques = [s.get("technique", "?") for s in stages_list]
            job = self.job_manager.create_job(
                trainer_type="sequential",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={"techniques": techniques, "num_stages": len(stages_list)},
            )

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_sequential(
                    stages=stages_list,
                    output_dir=output_dir,
                    base_model=base_model,
                    merge_between_stages=merge_between_stages,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "Sequential training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.job_status",
            description=(
                "Get real-time status of a training job. Returns current step, max steps, "
                "epoch, loss, learning rate, eval_loss, ETA, GPU memory, percent complete, "
                "and full result when completed. Use the job_id from finetune.train_async."
            ),
        )
        async def job_status(job_id: str) -> str:
            job = await self.job_manager.aget_job(job_id)
            if job is None:
                return json.dumps({"success": False, "error": f"Job not found: {job_id}"}, indent=2)
            return json.dumps({"success": True, **job.model_dump()}, indent=2)

        @self.mcp.tool(
            name="finetune.list_jobs",
            description=(
                "List all training jobs (running, completed, failed, cancelled). "
                "Optionally filter by status. Returns summary of each job."
            ),
        )
        async def list_jobs(
            status: Optional[str] = None,
            limit: int = 20,
        ) -> str:
            from shared.training_jobs import JobStatus as JS
            status_enum = JS(status) if status else None
            jobs = await self.job_manager.alist_jobs(status=status_enum, limit=limit)
            return json.dumps({
                "success": True,
                "count": len(jobs),
                "jobs": [j.model_dump() for j in jobs],
            }, indent=2)

        @self.mcp.tool(
            name="finetune.cancel_job",
            description=(
                "Cancel a running training job. The model checkpoint at the current step "
                "will be saved. The job status changes to 'cancelled'."
            ),
        )
        async def cancel_job(job_id: str) -> str:
            success = await self.job_manager.acancel_job(job_id)
            if not success:
                return json.dumps({
                    "success": False,
                    "error": f"Job not found or not running: {job_id}",
                }, indent=2)
            return json.dumps({
                "success": True,
                "job_id": job_id,
                "message": "Cancellation requested. Job will stop after current step.",
            }, indent=2)

        # Aliases expected by the frontend UI
        @self.mcp.tool(
            name="finetune.get_status",
            description=(
                "Get training job status. Without job_id returns all jobs; "
                "with job_id returns a single job's details."
            ),
        )
        async def get_status(job_id: Optional[str] = None) -> str:
            if job_id:
                job = await self.job_manager.aget_job(job_id)
                if job is None:
                    return json.dumps({"success": False, "error": f"Job not found: {job_id}"}, indent=2)
                return json.dumps({"success": True, **job.model_dump()}, indent=2)
            jobs = await self.job_manager.alist_jobs(limit=50)
            return json.dumps({
                "success": True,
                "jobs": [j.model_dump() for j in jobs],
            }, indent=2)

        @self.mcp.tool(
            name="finetune.cancel",
            description="Cancel a running training job by job_id.",
        )
        async def cancel(job_id: str) -> str:
            success = await self.job_manager.acancel_job(job_id)
            if not success:
                return json.dumps({
                    "success": False,
                    "error": f"Job not found or not running: {job_id}",
                }, indent=2)
            return json.dumps({
                "success": True,
                "job_id": job_id,
                "message": "Cancellation requested.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.delete_job",
            description=(
                "Delete a finished training job record by job_id. "
                "Active jobs must be cancelled first."
            ),
        )
        async def delete_job(job_id: str) -> str:
            success = await self.job_manager.adelete_job(job_id)
            if not success:
                return json.dumps({
                    "success": False,
                    "error": f"Job not found or still active: {job_id}",
                }, indent=2)
            return json.dumps({
                "success": True,
                "job_id": job_id,
                "message": "Job record deleted.",
            }, indent=2)

    # -- Test --
    def _register_test_tools(self):
        @self.mcp.tool(name="test.inference",
                       description="Run inference on prompts using a model")
        async def run_inference(
            prompts: List[str], model_path: str,
            adapter_path: Optional[str] = None,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
        ) -> str:
            result = await self.finetuner.run_inference(
                prompts=prompts, model_path=model_path,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="test.compare_models",
                       description="Compare base model vs fine-tuned model")
        async def compare_models(
            prompts: List[str], base_model_path: str,
            finetuned_adapter_path: str,
            max_new_tokens: int = 512,
        ) -> str:
            result = await self.finetuner.compare_models(
                prompts=prompts,
                base_model_path=base_model_path,
                finetuned_adapter_path=finetuned_adapter_path,
                max_new_tokens=max_new_tokens,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="test.vlm_inference",
            description="Run multimodal inference on structured messages using a local VLM.",
        )
        async def run_vlm_inference(
            messages: List[Dict[str, Any]],
            model_path: str,
            adapter_path: Optional[str] = None,
            max_new_tokens: int = 512,
            top_p: float = 0.9,
            top_k: int = 50,
        ) -> str:
            result = await self.finetuner.run_vlm_inference(
                messages=messages,
                model_path=model_path,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            return json.dumps(result, indent=2)

    # -- Validate --
    def _register_validate_tools(self):
        @self.mcp.tool(name="validate.model_info",
                       description="Get info about a local model or adapter")
        async def model_info(model_path: str) -> str:
            result = await self._invoke_dependency(self.finetuner.get_model_info, model_path)
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="validate.list_models",
                       description="List locally cached HuggingFace models")
        async def list_models(query: str = "") -> str:
            result = await self.finetuner.list_available_base_models(query=query)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="validate.search_models",
            description=(
                "Search HuggingFace Hub for models by query, task, and sort order. "
                "Returns model id, author, downloads, likes, tags, and library info."
            ),
        )
        async def search_models(
            query: str = "",
            task: str = "text-generation",
            sort: str = "downloads",
            limit: int = 20,
        ) -> str:
            result = await self.finetuner.search_huggingface_models(
                query=query, task=task, sort=sort, limit=limit,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="validate.recommend_models",
            description=(
                "Get curated model recommendations for a use case. "
                "Available use cases: general, low_memory, speed, quality, "
                "multilingual, indonesian."
            ),
        )
        async def recommend_models(use_case: ModelUseCase = "general") -> str:
            result = await self._invoke_dependency(
                self.finetuner.get_recommended_models, use_case=use_case,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="validate.schema",
            description=(
                "Validate that a dataset file has the correct columns for a "
                "training technique (sft, dpo, grpo, kto, vlm_sft). Returns success, "
                "detected technique, expected columns, and any mismatches."
            ),
        )
        async def validate_schema(
            dataset_path: str,
            technique: SchemaTechniqueName = "sft",
        ) -> str:
            meta = await self.dataset_service.info(dataset_path)
            if not meta.get("success"):
                return json.dumps(meta, indent=2)

            detected = meta["metadata"].get("technique")
            columns = meta["metadata"].get("columns", [])

            expected = {
                "sft": {"instruction", "output"},
                "dpo": {"prompt", "chosen", "rejected"},
                "grpo": {"prompt", "responses", "rewards"},
                "kto": {"prompt", "completion", "label"},
                "vlm_sft": {"messages"},
            }
            required = expected.get(technique, set())
            col_set = set(columns)
            missing = required - col_set

            return json.dumps({
                "success": len(missing) == 0,
                "technique_requested": technique,
                "technique_detected": detected,
                "columns": columns,
                "missing_columns": sorted(missing) if missing else [],
                "row_count": meta["metadata"].get("row_count", 0),
            }, indent=2)

        @self.mcp.tool(
            name="validate.data_quality",
            description=(
                "Quick data quality check on a dataset file. Returns row count, "
                "column stats, empty field counts, and average text lengths."
            ),
        )
        async def validate_data_quality(
            dataset_path: str,
        ) -> str:
            meta = await self.dataset_service.info(dataset_path)
            if not meta.get("success"):
                return json.dumps(meta, indent=2)

            stats = await self.dataset_service.sample_text_stats(dataset_path)
            row_count = meta["metadata"].get("row_count", 0)

            result = {
                "success": True,
                "row_count": row_count,
                "columns": meta["metadata"].get("columns", []),
                "technique": meta["metadata"].get("technique"),
            }
            if stats.get("success"):
                result["avg_text_length"] = stats.get("avg_length", 0)
                result["p95_text_length"] = stats.get("p95_length", 0)
                result["empty_count"] = stats.get("empty_count", 0)

            return json.dumps(result, indent=2)

    # -- Host --
    def _register_host_tools(self):
        @self.mcp.tool(
            name="host.deploy_mcp",
            description=(
                "Deploy a fine-tuned model as an MCP tool server. "
                "Use model_path from finetune.train* results as adapter_path. "
                "Returns deployment_id and endpoint URL."
            ),
        )
        async def deploy_mcp(
            model_path: str, adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
            quantization: Optional[str] = None,
        ) -> str:
            config = HostingConfig(
                model_path=model_path, adapter_path=adapter_path,
                name=name,
                port=port, quantization=quantization,
            )
            return json.dumps(await self.hoster.deploy_as_mcp(config), indent=2)

        @self.mcp.tool(
            name="host.deploy_vlm_mcp",
            description=(
                "Deploy a vision-language model as an MCP tool server. "
                "Returns deployment_id and endpoint URL."
            ),
        )
        async def deploy_vlm_mcp(
            model_path: str,
            adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
        ) -> str:
            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                name=name,
                port=port,
                modality="vision-language",
            )
            return json.dumps(await self.hoster.deploy_vlm_as_mcp(config), indent=2)

        @self.mcp.tool(
            name="host.deploy_api",
            description=(
                "Deploy a fine-tuned model as a REST API with /generate endpoint. "
                "Use model_path from finetune.train* results as adapter_path. "
                "Returns deployment_id and endpoint URL."
            ),
        )
        async def deploy_api(
            model_path: str, adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
            quantization: Optional[str] = None,
        ) -> str:
            config = HostingConfig(
                model_path=model_path, adapter_path=adapter_path,
                name=name,
                port=port, quantization=quantization,
            )
            return json.dumps(await self.hoster.deploy_as_api(config), indent=2)

        @self.mcp.tool(
            name="host.deploy_vlm_api",
            description=(
                "Deploy a vision-language model as a REST API with a /generate_vlm endpoint. "
                "Returns deployment_id and endpoint URL."
            ),
        )
        async def deploy_vlm_api(
            model_path: str,
            adapter_path: Optional[str] = None,
            name: Optional[str] = None,
            port: int = 8001,
        ) -> str:
            config = HostingConfig(
                model_path=model_path,
                adapter_path=adapter_path,
                name=name,
                port=port,
                modality="vision-language",
            )
            return json.dumps(await self.hoster.deploy_vlm_as_api(config), indent=2)

        @self.mcp.tool(name="host.list_deployments",
                       description="List running model deployments")
        async def list_deployments() -> str:
            return json.dumps(await self.hoster.list_deployments(), indent=2)

        @self.mcp.tool(name="host.stop",
                       description="Stop a running deployment")
        async def stop_deployment(deployment_id: str) -> str:
            return json.dumps(await self.hoster.stop_deployment(deployment_id), indent=2)

        @self.mcp.tool(
            name="host.delete_deployment",
            description="Delete a deployment record. Stops the deployment first if it is still running.",
        )
        async def delete_deployment(deployment_id: str) -> str:
            return json.dumps(await self.hoster.delete_deployment(deployment_id), indent=2)

        @self.mcp.tool(
            name="host.health",
            description="Health check on a running deployment. Returns status, uptime, endpoint.",
        )
        async def host_health(deployment_id: str) -> str:
            return json.dumps(await self.hoster.health_check(deployment_id), indent=2)

        @self.mcp.tool(
            name="host.chat",
            description=(
                "Chat with a deployed or local text fine-tuned model. "
                "IMPORTANT: After any finetune.train* with deploy=True succeeds, "
                "offer to use this tool so the user can test the model interactively. "
                "Pass deployment_id to use an active deployment directly, or pass the endpoint "
                "from the deployment result, or a model_path for direct loading. "
                "Use conversation_id to maintain multi-turn context across calls. "
                "Also share the deployment's chat_command for standalone CLI access."
            ),
        )
        async def chat_with_model(
            message: str,
            deployment_id: Optional[str] = None,
            endpoint: Optional[str] = None,
            model_path: Optional[str] = None,
            adapter_path: Optional[str] = None,
            conversation_id: Optional[str] = None,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            top_k: int = 50,
            system_prompt: Optional[str] = None,
            prefer_runtime_metrics: bool = False,
        ) -> str:
            session_result = await self._get_or_create_text_chat_session(
                deployment_id=deployment_id,
                endpoint=endpoint,
                model_path=model_path,
                adapter_path=adapter_path,
                conversation_id=conversation_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                system_prompt=system_prompt,
                prefer_runtime_metrics=prefer_runtime_metrics,
            )
            if not session_result.get("success"):
                return json.dumps(session_result, indent=2)

            cid = session_result["conversation_id"]
            session = session_result["session"]

            if hasattr(session, "send_message_result"):
                result = await session.send_message_result(message)
            else:
                result = {
                    "response": await session.send_message(message),
                    "metrics": None,
                    "usage": None,
                    "model_id": None,
                }
            await self._persist_conversation_exchange(
                conversation_id=cid,
                user_content=message,
                assistant_content=result["response"],
            )
            return json.dumps(
                {
                    "success": True,
                    "conversation_id": cid,
                    "deployment_id": deployment_id,
                    "response": result["response"],
                    "turns": session.get_info()["turns"],
                    "metrics": result.get("metrics"),
                    "usage": result.get("usage"),
                    "model_id": result.get("model_id"),
                },
                indent=2,
            )

        @self.mcp.tool(
            name="host.chat_vlm",
            description=(
                "Chat with a deployed or local vision-language model using structured multimodal messages. "
                "Pass deployment_id to reuse an active VLM deployment, or pass endpoint/model_path directly. "
                "Use conversation_id to maintain multi-turn context across calls."
            ),
        )
        async def chat_with_vlm(
            messages: Union[str, List[Dict[str, Any]]],
            deployment_id: Optional[str] = None,
            endpoint: Optional[str] = None,
            model_path: Optional[str] = None,
            adapter_path: Optional[str] = None,
            conversation_id: Optional[str] = None,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            top_k: int = 50,
            system_prompt: Optional[str] = None,
            prefer_runtime_metrics: bool = False,
        ) -> str:
            from hosting_pipeline.services.chat_service import ChatSession

            parsed_messages = messages
            if isinstance(parsed_messages, str):
                parsed_messages = json.loads(parsed_messages)
            if not isinstance(parsed_messages, list) or not parsed_messages:
                return json.dumps(
                    {"success": False, "error": "messages must be a non-empty list"},
                    indent=2,
                )

            cid = conversation_id or str(uuid.uuid4())[:8]
            persisted_conversation = None

            if cid in self._chat_sessions:
                session = self._chat_sessions[cid]
                if session.get_info().get("modality") != "vision-language":
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Conversation is text-only. Use host.chat for this conversation.",
                            "conversation_id": cid,
                        },
                        indent=2,
                    )
                if hasattr(session, "update_generation_config"):
                    session.update_generation_config(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
            else:
                provider = None
                inference_service = None
                resolved_endpoint = endpoint
                resolved_model_path = model_path
                resolved_adapter_path = adapter_path
                resolved_api_path = "/generate_vlm"

                if conversation_id:
                    persisted_conversation = await self._persistence.get_conversation(cid)
                    if persisted_conversation:
                        deployment_id = deployment_id or persisted_conversation.get("deployment_id")
                        resolved_endpoint = resolved_endpoint or persisted_conversation.get("endpoint")
                        resolved_model_path = (
                            resolved_model_path or persisted_conversation.get("model_path")
                        )
                        resolved_adapter_path = (
                            resolved_adapter_path or persisted_conversation.get("adapter_path")
                        )
                        system_prompt = system_prompt or persisted_conversation.get("system_prompt")

                if deployment_id:
                    deployment = self.hoster.get_deployment(deployment_id)
                    if deployment is None:
                        deployment = await self._persistence.get_deployment(deployment_id)
                        if deployment is None:
                            return json.dumps(
                                {"success": False, "error": f"Deployment {deployment_id} not found"},
                                indent=2,
                            )
                    deployment_host = deployment.get("host")
                    if deployment_host in {"0.0.0.0", "::"}:
                        deployment_host = "127.0.0.1"
                    deployment_endpoint = (
                        f"http://{deployment_host}:{deployment['port']}"
                        if deployment.get("transport") == "http"
                        else None
                    )
                    if deployment.get("modality") != "vision-language":
                        return json.dumps(
                            {
                                "success": False,
                                "error": "Deployment is text-only. Use host.chat instead.",
                            },
                            indent=2,
                        )
                    if (
                        deployment.get("type") == "api"
                        and deployment_endpoint
                        and not prefer_runtime_metrics
                    ):
                        resolved_endpoint = deployment_endpoint
                        resolved_api_path = deployment.get("api_path") or "/generate_vlm"
                    else:
                        resolved_model_path = deployment.get("model_path")
                        resolved_adapter_path = deployment.get("adapter_path")
                        inference_service = deployment.get("inference_service")
                        provider = deployment.get("provider")

                config = ChatConfig(
                    endpoint=resolved_endpoint,
                    model_path=resolved_model_path,
                    adapter_path=resolved_adapter_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    system_prompt=system_prompt,
                    modality="vision-language",
                    api_path=resolved_api_path,
                )
                session = ChatSession(
                    config,
                    provider=provider,
                    inference_service=inference_service,
                )
                await session.initialize()
                if persisted_conversation:
                    await self._restore_persisted_conversation(session, cid)
                self._chat_sessions[cid] = session
                await self._persist_conversation_metadata(
                    conversation_id=cid,
                    deployment_id=deployment_id,
                    modality="vision-language",
                    endpoint=resolved_endpoint,
                    model_path=resolved_model_path,
                    adapter_path=resolved_adapter_path,
                    system_prompt=system_prompt,
                )

            if hasattr(session, "send_messages_result"):
                result = await session.send_messages_result(parsed_messages)
            else:
                result = {
                    "response": await session.send_messages(parsed_messages),
                    "metrics": None,
                    "usage": None,
                    "model_id": None,
                }
            await self._persist_conversation_exchange(
                conversation_id=cid,
                user_content=parsed_messages,
                assistant_content=result["response"],
            )
            return json.dumps(
                {
                    "success": True,
                    "conversation_id": cid,
                    "deployment_id": deployment_id,
                    "response": result["response"],
                    "turns": session.get_info()["turns"],
                    "modality": "vision-language",
                    "metrics": result.get("metrics"),
                    "usage": result.get("usage"),
                    "model_id": result.get("model_id"),
                },
                indent=2,
            )

        @self.mcp.tool(
            name="host.list_conversations",
            description=(
                "List persisted deployment chat conversations. "
                "Optionally filter by deployment_id or modality."
            ),
        )
        async def list_host_conversations(
            deployment_id: Optional[str] = None,
            modality: Optional[str] = None,
            limit: int = 20,
        ) -> str:
            conversations = await self._persistence.list_conversations(
                deployment_id=deployment_id,
                modality=modality,
                limit=limit,
            )
            return json.dumps(
                {
                    "success": True,
                    "count": len(conversations),
                    "conversations": conversations,
                },
                indent=2,
            )

        @self.mcp.tool(
            name="host.get_conversation",
            description="Load a persisted deployment chat conversation, including messages.",
        )
        async def get_host_conversation(conversation_id: str) -> str:
            conversation = await self._persistence.get_conversation(conversation_id)
            if conversation is None:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Conversation not found: {conversation_id}",
                    },
                    indent=2,
                )
            return json.dumps({"success": True, **conversation}, indent=2)

        @self.mcp.tool(
            name="host.rename_conversation",
            description="Rename a persisted deployment chat conversation.",
        )
        async def rename_host_conversation(conversation_id: str, title: str) -> str:
            normalized = " ".join(title.split()).strip()
            if not normalized:
                return json.dumps(
                    {"success": False, "error": "title must be a non-empty string"},
                    indent=2,
                )
            success = await self._persistence.set_conversation_title(conversation_id, normalized)
            if not success:
                return json.dumps(
                    {"success": False, "error": f"Conversation not found: {conversation_id}"},
                    indent=2,
                )
            return json.dumps(
                {"success": True, "conversation_id": conversation_id, "title": normalized},
                indent=2,
            )

        @self.mcp.tool(
            name="host.delete_conversation",
            description="Delete a persisted deployment chat conversation and its stored messages.",
        )
        async def delete_host_conversation(conversation_id: str) -> str:
            success = await self._persistence.delete_conversation(conversation_id)
            if not success:
                return json.dumps(
                    {"success": False, "error": f"Conversation not found: {conversation_id}"},
                    indent=2,
                )
            self._chat_sessions.pop(conversation_id, None)
            return json.dumps(
                {"success": True, "conversation_id": conversation_id},
                indent=2,
            )

    # -- Workflow --
    def _register_workflow_tools(self):
        class WorkflowExecutionError(RuntimeError):
            def __init__(self, message: str, result: Optional[Dict[str, Any]] = None):
                super().__init__(message)
                self.result = result

        def _coerce_file_paths(
            file_path: Optional[str] = None,
            file_paths: Optional[Union[str, List[str]]] = None,
        ) -> tuple[Optional[str], Optional[List[str]]]:
            resolved: List[str] = []
            if file_path and file_path.strip():
                resolved.append(file_path.strip())
            if isinstance(file_paths, str):
                try:
                    parsed = json.loads(file_paths)
                    if isinstance(parsed, list):
                        resolved.extend(
                            item.strip() for item in parsed
                            if isinstance(item, str) and item.strip()
                        )
                    elif isinstance(parsed, str) and parsed.strip():
                        resolved.append(parsed.strip())
                except (json.JSONDecodeError, ValueError):
                    resolved.extend(
                        part.strip() for part in file_paths.splitlines()
                        if part.strip()
                    )
            elif isinstance(file_paths, list):
                resolved.extend(
                    item.strip() for item in file_paths
                    if isinstance(item, str) and item.strip()
                )

            unique: List[str] = []
            seen = set()
            for item in resolved:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)

            if not unique:
                raise ValueError("Provide file_path or file_paths")
            if len(unique) == 1:
                return unique[0], None
            return None, unique

        def _serialize_workflow_job(job) -> Dict[str, Any]:
            data = job.model_dump()
            data["steps"] = data.get("config_summary", {}).get("steps", [])
            return data

        @self.mcp.tool(name="workflow.full_pipeline",
                       description="End-to-end pipeline over one or more documents: Extract -> Generate -> Clean -> Normalize -> Evaluate -> Filter -> Train -> Test -> Host")
        async def full_pipeline(
            file_path: Optional[str] = None,
            file_paths: Optional[Union[str, List[str]]] = None,
            technique: TechniqueName = "sft",
            output_dir: str = "./output",
            quality_threshold: float = 0.7,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            push_to_hub: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
            quantization: Optional[str] = None,
        ) -> str:
            resolved_file_path, resolved_file_paths = _coerce_file_paths(file_path, file_paths)
            result = await self.orchestrator.full_pipeline(
                file_path=resolved_file_path,
                file_paths=resolved_file_paths,
                technique=technique,
                output_dir=output_dir,
                quality_threshold=quality_threshold,
                base_model=base_model,
                num_epochs=num_epochs,
                use_lora=use_lora,
                push_to_hub=push_to_hub,
                deploy=deploy,
                deploy_port=deploy_port,
                quantization=quantization,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="workflow.generate_and_evaluate",
                       description="Generate and score a dataset from one or more documents.")
        async def generate_and_evaluate(
            file_path: Optional[str] = None,
            file_paths: Optional[Union[str, List[str]]] = None,
            technique: TechniqueName = "sft",
            quality_threshold: float = 0.7,
        ) -> str:
            resolved_file_path, resolved_file_paths = _coerce_file_paths(file_path, file_paths)
            result = await self.orchestrator.generate_and_evaluate(
                file_path=resolved_file_path,
                file_paths=resolved_file_paths,
                technique=technique,
                quality_threshold=quality_threshold,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="workflow.curriculum_pipeline",
            description=(
                "Full curriculum learning pipeline from raw documents to a compared model. "
                "Runs: Extract -> Generate -> Clean -> Normalize -> Evaluate -> Filter -> "
                "Curriculum Train (staged easy->hard) -> Compare vs base model -> (optional) Deploy. "
                "Accepts one or more Markdown/PDF/text files. "
                "The evaluate step scores every row with weighted_score so curriculum "
                "training always receives a pre-scored dataset -- no double work."
            ),
        )
        async def curriculum_pipeline(
            file_paths: Union[str, List[str]],
            output_dir: str,
            technique: TechniqueName = "sft",
            quality_threshold: float = 0.6,
            base_model: Optional[str] = None,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            difficulty_order: DifficultyOrder = "easy_first",
            use_lora: bool = True,
            lora_r: int = 8,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            # Accept a JSON-encoded list or a bare string (single file)
            if isinstance(file_paths, str):
                try:
                    parsed = json.loads(file_paths)
                    file_paths = parsed if isinstance(parsed, list) else [parsed]
                except (json.JSONDecodeError, ValueError):
                    file_paths = [file_paths]
            result = await self.orchestrator.curriculum_pipeline(
                file_paths=file_paths,
                output_dir=output_dir,
                technique=technique,
                quality_threshold=quality_threshold,
                base_model=base_model,
                num_stages=num_stages,
                num_epochs_per_stage=num_epochs_per_stage,
                difficulty_order=difficulty_order,
                use_lora=use_lora,
                lora_r=lora_r,
                deploy=deploy,
                deploy_port=deploy_port,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="workflow.compare_flat_vs_curriculum",
            description=(
                "Compare flat SFT vs curriculum SFT on the same dataset. "
                "Trains both approaches and provides side-by-side comparison "
                "using test.compare_models. Returns structured results with "
                "training metrics and qualitative output comparisons."
            ),
        )
        async def compare_flat_vs_curriculum(
            dataset_path: str,
            output_dir: str = "./output/comparison",
            base_model: Optional[str] = None,
            num_epochs_flat: int = 3,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            difficulty_order: DifficultyOrder = "easy_first",
            score_column: str = "weighted_score",
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            load_in_4bit: bool = True,
            learning_rate: float = 2e-4,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            gradient_checkpointing: bool = False,
            max_seq_length: int = 2048,
            warmup_ratio: float = 0.0,
            test_data_path: Optional[str] = None,
            test_prompts: Optional[str] = None,
        ) -> str:
            parsed_prompts = None
            if test_prompts:
                parsed_prompts = (
                    json.loads(test_prompts)
                    if isinstance(test_prompts, str)
                    else test_prompts
                )
            result = await self.orchestrator.compare_flat_vs_curriculum(
                dataset_path=dataset_path,
                output_dir=output_dir,
                base_model=base_model,
                num_epochs_flat=num_epochs_flat,
                num_stages=num_stages,
                num_epochs_per_stage=num_epochs_per_stage,
                difficulty_order=difficulty_order,
                score_column=score_column,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                load_in_4bit=load_in_4bit,
                learning_rate=learning_rate,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                max_seq_length=max_seq_length,
                warmup_ratio=warmup_ratio,
                test_data_path=test_data_path,
                test_prompts=parsed_prompts,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="workflow.run_pipeline",
            description=(
                "Execute a sequence of MCP tools server-side in a single call. "
                "Avoids back-and-forth round-trips between client and server. "
                "Each step specifies a tool name and params. Use '$prev.key' in "
                "params to reference the previous step's output (e.g., "
                "'$prev.model_path' passes the model_path from the prior step). "
                "Stops on first failure and returns partial results. "
                "Set dry_run=True to validate the pipeline without executing. "
                "Example steps: "
                '[{"tool":"system.preflight_check","params":{"model_name":"meta-llama/Llama-3.2-3B-Instruct"}},'
                '{"tool":"finetune.train","params":{"dataset_path":"data.jsonl"}},'
                '{"tool":"test.inference","params":{"model_path":"$prev.model_path"}}]'
            ),
        )
        async def run_pipeline(steps: str, dry_run: bool = False) -> str:
            from shared.pipeline_executor import PipelineExecutor, PipelineStep

            parsed_steps = [PipelineStep(**s) for s in json.loads(steps)]
            executor = PipelineExecutor(self.mcp._tools)
            result = await executor.execute(parsed_steps, dry_run=dry_run)
            return result.model_dump_json(indent=2)

        @self.mcp.tool(
            name="workflow.run_pipeline_async",
            description=(
                "Run a custom workflow pipeline in the background and return a job_id. "
                "Poll workflow.job_status(job_id) or workflow.list_jobs()."
            ),
        )
        async def run_pipeline_async(
            steps: str,
            dry_run: bool = False,
            output_dir: str = "./output/workflow_jobs/custom",
        ) -> str:
            from shared.pipeline_executor import PipelineExecutor, PipelineStep

            parsed_steps = [PipelineStep(**s) for s in json.loads(steps)]
            step_names = [step.tool for step in parsed_steps]
            job = self.workflow_job_manager.create_job(
                trainer_type="workflow_custom",
                base_model="workflow.run_pipeline",
                output_dir=output_dir,
                config_summary={
                    "steps": step_names,
                    "dry_run": dry_run,
                    "pipeline_type": "custom",
                },
            )

            async def _run_workflow(extra_callbacks=None):
                cancel_event = self.workflow_job_manager.get_cancel_event(job.job_id)
                executor = PipelineExecutor(self.mcp._tools)

                async def _step_update(phase: str, step_index: int, total_steps: int, step, **kwargs):
                    current_step = step_index if phase == "start" else step_index + 1
                    self.workflow_job_manager.update_progress(
                        job.job_id,
                        current_stage=step.tool,
                        current_step=current_step,
                        max_steps=total_steps,
                        percent_complete=round(current_step / max(total_steps, 1) * 100, 1),
                    )

                async def _tool_runner(step, resolved_params, tool_func):
                    intercepted = await self._run_workflow_training_tool(
                        step.tool,
                        resolved_params,
                        extra_callbacks=extra_callbacks,
                    )
                    if intercepted is not None:
                        return intercepted
                    return await tool_func(**resolved_params)

                result = await executor.execute(
                    parsed_steps,
                    dry_run=dry_run,
                    step_callback=_step_update,
                    cancel_event=cancel_event,
                    tool_runner=_tool_runner,
                )
                payload = result.model_dump()
                if not payload.get("success"):
                    if cancel_event is not None and cancel_event.is_set():
                        return payload
                    raise WorkflowExecutionError(payload.get("error", "Workflow pipeline failed"), payload)
                return payload

            await self.workflow_job_manager.start_job(job.job_id, _run_workflow)
            return json.dumps({
                "success": True,
                "job_id": job.job_id,
                "status": "running",
                "message": "Workflow pipeline started. Use workflow.job_status to monitor progress.",
            }, indent=2)

        @self.mcp.tool(
            name="workflow.full_pipeline_async",
            description=(
                "Run the full pipeline in the background for one or more documents. "
                "Returns a job_id immediately."
            ),
        )
        async def full_pipeline_async(
            file_path: Optional[str] = None,
            file_paths: Optional[Union[str, List[str]]] = None,
            technique: TechniqueName = "sft",
            output_dir: str = "./output",
            quality_threshold: float = 0.7,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            push_to_hub: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
            quantization: Optional[str] = None,
        ) -> str:
            resolved_file_path, resolved_file_paths = _coerce_file_paths(file_path, file_paths)
            resolved_paths = resolved_file_paths or ([resolved_file_path] if resolved_file_path else [])
            job = self.workflow_job_manager.create_job(
                trainer_type="workflow_full",
                base_model=base_model or "meta-llama/Llama-3.2-3B-Instruct",
                output_dir=output_dir,
                config_summary={
                    "steps": ["generate", "clean", "normalize", "evaluate", "filter", "export", "train", "test", "deploy"],
                    "pipeline_type": "full",
                    "technique": technique,
                    "file_count": len(resolved_paths),
                    "use_lora": use_lora,
                    "push_to_hub": push_to_hub,
                    "deploy": deploy,
                    "quantization": quantization,
                },
            )

            async def _run_workflow(extra_callbacks=None):
                cancel_event = self.workflow_job_manager.get_cancel_event(job.job_id)

                async def _progress_update(**kwargs):
                    self.workflow_job_manager.update_progress(job.job_id, **kwargs)

                result = await self.orchestrator.full_pipeline(
                    file_path=resolved_file_path,
                    file_paths=resolved_file_paths,
                    technique=technique,
                    output_dir=output_dir,
                    quality_threshold=quality_threshold,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    use_lora=use_lora,
                    push_to_hub=push_to_hub,
                    deploy=deploy,
                    deploy_port=deploy_port,
                    quantization=quantization,
                    progress_callback=_progress_update,
                    cancel_event=cancel_event,
                    extra_callbacks=extra_callbacks,
                )
                if not result.get("success"):
                    if cancel_event is not None and cancel_event.is_set():
                        return result
                    raise WorkflowExecutionError(result.get("error", "Full pipeline failed"), result)
                return result

            await self.workflow_job_manager.start_job(job.job_id, _run_workflow)
            return json.dumps({
                "success": True,
                "job_id": job.job_id,
                "status": "running",
                "message": "Full pipeline started. Use workflow.job_status to monitor progress.",
            }, indent=2)

        @self.mcp.tool(
            name="workflow.job_status",
            description="Get the current status of an async workflow job.",
        )
        async def workflow_job_status(job_id: str) -> str:
            job = await self.workflow_job_manager.aget_job(job_id)
            if job is None:
                return json.dumps({"success": False, "error": f"Job not found: {job_id}"}, indent=2)
            return json.dumps({"success": True, **_serialize_workflow_job(job)}, indent=2)

        @self.mcp.tool(
            name="workflow.list_jobs",
            description="List workflow jobs started with workflow.run_pipeline_async or workflow.full_pipeline_async.",
        )
        async def workflow_list_jobs(status: Optional[str] = None, limit: int = 20) -> str:
            from shared.training_jobs import JobStatus as JS

            status_enum = JS(status) if status else None
            jobs = await self.workflow_job_manager.alist_jobs(status=status_enum, limit=limit)
            return json.dumps({
                "success": True,
                "count": len(jobs),
                "jobs": [_serialize_workflow_job(job) for job in jobs],
            }, indent=2)

        @self.mcp.tool(
            name="workflow.cancel_job",
            description="Cancel a running workflow job.",
        )
        async def workflow_cancel_job(job_id: str) -> str:
            success = self.workflow_job_manager.cancel_job(job_id)
            if not success:
                return json.dumps({
                    "success": False,
                    "error": f"Job not found or not running: {job_id}",
                }, indent=2)
            return json.dumps({
                "success": True,
                "job_id": job_id,
                "message": "Cancellation requested. Workflow stops after the current stage.",
            }, indent=2)

        @self.mcp.tool(
            name="workflow.guided_pipeline",
            description=(
                "Describe what you want to do in plain English. Returns the exact "
                "sequence of MCP tool calls with parameters to execute your goal. "
                "Use this FIRST when you don't know which tools to call. "
                "This is a pure planning tool — it does NOT execute anything. "
                "Take the returned steps and either pass them to workflow.run_pipeline "
                "or call the tools individually."
            ),
        )
        async def guided_pipeline(
            goal: str,
            file_path: Optional[str] = None,
            base_model: Optional[str] = None,
        ) -> str:
            from shared.workflow_planner import WorkflowPlanner

            planner = WorkflowPlanner()
            plan = planner.plan(goal, file_path=file_path, base_model=base_model)
            return json.dumps(plan, indent=2)

    # -- Orchestration --
    def _register_orchestration_tools(self):
        @self.mcp.tool(
            name="orchestration.generate_problems",
            description="Generate synthetic orchestration tasks for a domain",
        )
        async def generate_problems(
            domain_description: str,
            num_problems: int = 50,
            tool_descriptions: Optional[List[Dict]] = None,
        ) -> str:
            result = await self.orchestration_data_service.generate_problems(
                domain_description=domain_description,
                num_problems=num_problems,
                tool_descriptions=tool_descriptions or self._default_orchestration_tool_descriptions(),
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="orchestration.collect_trajectories",
            description="Run problems through agent, record trajectories with cost/latency",
        )
        async def collect_trajectories(
            problems: List[Dict],
            n_per_problem: int = 4,
        ) -> str:
            agent = self._build_internal_orchestration_agent()
            result = await self.orchestration_data_service.collect_trajectories(
                problems=problems, agent=agent, n_per_problem=n_per_problem,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="orchestration.build_training_data",
            description="Score trajectories and convert to SFT/DPO/GRPO training format",
        )
        async def build_training_data(
            collected: List[Dict],
            format: OrchestrationTrainingFormat = "sft",
            tool_descriptions: Optional[List[Dict]] = None,
            cost_budget: float = 1.0,
            time_budget: float = 60.0,
        ) -> str:
            result = await self.orchestration_data_service.build_training_data(
                collected=collected,
                format=format,
                tool_descriptions=tool_descriptions or self._default_orchestration_tool_descriptions(),
                cost_budget=cost_budget,
                time_budget=time_budget,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="orchestration.train_orchestrator",
            description="Full pipeline: problems -> trajectories -> rewards -> train -> deploy",
        )
        async def train_orchestrator(
            domain_description: str,
            num_problems: int = 50,
            n_per_problem: int = 4,
            output_dir: str = "./output/orchestrator",
            output_format: OrchestrationTrainingFormat = "sft",
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            deploy: bool = False,
            deploy_port: int = 8002,
            training_data: Optional[List[Dict]] = None,
        ) -> str:
            result = await self.orchestrator.train_orchestrator(
                domain_description=domain_description,
                agent=self._build_internal_orchestration_agent() if training_data is None else None,
                num_problems=num_problems,
                n_per_problem=n_per_problem,
                output_dir=output_dir,
                output_format=output_format,
                base_model=base_model,
                num_epochs=num_epochs,
                deploy=deploy,
                deploy_port=deploy_port,
                tool_descriptions=self._default_orchestration_tool_descriptions(),
                training_data=training_data,
            )
            return json.dumps(result, indent=2)

    # -- Judge (Advanced LLM-as-a-Judge) --
    def _register_judge_tools(self):
        @self.mcp.tool(
            name="judge.evaluate",
            description="Run a single LLM-as-a-judge evaluation on one sample with custom criteria/rubric",
        )
        async def judge_evaluate(
            question: str,
            generated: str,
            reference: Optional[str] = None,
            generated_b: Optional[str] = None,
            judge_type: str = "pointwise",
            judge_model: str = "gpt-4o",
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
        ) -> str:
            result = await self.advanced_judge.evaluate_single(
                question=question, generated=generated,
                reference=reference, generated_b=generated_b,
                judge_type=judge_type, judge_model=judge_model,
                criteria=criteria, rubric=rubric,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.evaluate_multi",
            description="Run multiple LLM judges in parallel on one sample and aggregate scores",
        )
        async def judge_evaluate_multi(
            question: str,
            generated: str,
            reference: Optional[str] = None,
            generated_b: Optional[str] = None,
            judge_type: str = "pointwise",
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
            aggregation: str = "mean",
        ) -> str:
            result = await self.advanced_judge.evaluate_multi_judge(
                question=question, generated=generated,
                reference=reference, generated_b=generated_b,
                judge_type=judge_type, judges=judges,
                criteria=criteria, rubric=rubric,
                aggregation=aggregation,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.evaluate_batch",
            description="Run batch evaluation with multi-judge support and custom criteria",
        )
        async def judge_evaluate_batch(
            test_data: List[Dict],
            judge_type: str = "pointwise",
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
            aggregation: str = "mean",
        ) -> str:
            result = await self.advanced_judge.evaluate_batch(
                test_data=test_data, judge_type=judge_type,
                judges=judges, criteria=criteria,
                rubric=rubric, aggregation=aggregation,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.compare_pair",
            description="Pairwise comparison: which of two outputs is better?",
        )
        async def judge_compare_pair(
            question: str,
            generated_a: str,
            generated_b: str,
            reference: Optional[str] = None,
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None,
        ) -> str:
            result = await self.advanced_judge.evaluate_multi_judge(
                question=question,
                generated=generated_a,
                generated_b=generated_b,
                reference=reference,
                judge_type="pairwise",
                judges=judges,
                criteria=criteria,
                aggregation="mean",
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.evaluate_vlm",
            description="Run a single multimodal judge evaluation using text and image message blocks.",
        )
        async def judge_evaluate_vlm(
            messages: List[Dict[str, Any]],
            generated: str,
            reference: Optional[str] = None,
            judge_model: str = "gpt-4o",
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
        ) -> str:
            result = await self.advanced_judge.evaluate_vlm_single(
                messages=messages,
                generated=generated,
                reference=reference,
                judge_model=judge_model,
                criteria=criteria,
                rubric=rubric,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.compare_vlm",
            description="Compare two candidate answers against the same multimodal input.",
        )
        async def judge_compare_vlm(
            messages: List[Dict[str, Any]],
            generated_a: str,
            generated_b: str,
            reference: Optional[str] = None,
            judge_model: str = "gpt-4o",
            criteria: Optional[List[Dict]] = None,
        ) -> str:
            result = await self.advanced_judge.compare_vlm(
                messages=messages,
                generated_a=generated_a,
                generated_b=generated_b,
                reference=reference,
                judge_model=judge_model,
                criteria=criteria,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.evaluate_vlm_batch",
            description="Run batch multimodal judge evaluation on rows with canonical messages plus generated outputs.",
        )
        async def judge_evaluate_vlm_batch(
            test_data: List[Dict],
            judge_model: str = "gpt-4o",
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
        ) -> str:
            result = await self.advanced_judge.evaluate_vlm_batch(
                test_data=test_data,
                judge_model=judge_model,
                criteria=criteria,
                rubric=rubric,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.list_types",
            description="List available judge evaluation types (pointwise, pairwise, reference_free, rubric)",
        )
        async def judge_list_types() -> str:
            result = await self._invoke_dependency(self.advanced_judge.list_judge_types)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.export",
            description="Export judge evaluation results as JSONL or JSON",
        )
        async def judge_export(
            results: List[Dict],
            output_path: str,
            format: ResultExportFormat = "jsonl",
        ) -> str:
            result = await self.advanced_judge.export_results(
                results=results, output_path=output_path, format=format,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.create_rubric",
            description="Validate a rubric definition and return the parsed result",
        )
        async def judge_create_rubric(
            name: str,
            criteria: List[Dict],
            description: str = "",
        ) -> str:
            from model_evaluator_pipeline.judges.models import JudgeCriterion, JudgeRubric
            try:
                rubric = JudgeRubric(
                    name=name, description=description,
                    criteria=[JudgeCriterion(**c) for c in criteria],
                )
                return json.dumps({
                    "success": True,
                    "rubric": rubric.model_dump(),
                }, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, indent=2)

    # -- FT Evaluator (Domain Knowledge Judge) --
    def _register_ft_evaluator_tools(self):
        @self.mcp.tool(
            name="ft_eval.single",
            description="Domain knowledge PASS/FAIL evaluation on one sample (single judge)",
        )
        async def ft_eval_single(
            instruction: str,
            generated: str,
            reference: str,
            judge_model: Optional[str] = None,
            ksmi_label: Optional[str] = None,
        ) -> str:
            verdict = await self.ft_evaluator.evaluate_single(
                instruction=instruction,
                generated=generated,
                reference=reference,
                judge_model=judge_model,
                ksmi_label=ksmi_label,
            )
            return json.dumps(verdict.model_dump(), indent=2)

        @self.mcp.tool(
            name="ft_eval.batch",
            description="Batch domain knowledge evaluation with multi-judge + KSMI labels",
        )
        async def ft_eval_batch(
            test_data: List[Dict],
            judge_models: Optional[List[str]] = None,
        ) -> str:
            result = await self.ft_evaluator.evaluate_batch(
                test_data=test_data,
                judge_models=judge_models,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="ft_eval.summary",
            description="Compute stakeholder summary (pass rate, failure types, severity, KSMI breakdown)",
        )
        async def ft_eval_summary(results: List[Dict]) -> str:
            from model_evaluator_pipeline.models.ft_evaluator import FTEvalResult
            parsed = [FTEvalResult(**r) for r in results]
            summary = await self._invoke_dependency(self.ft_evaluator.compute_summary, parsed)
            return json.dumps({"success": True, "summary": summary.model_dump()}, indent=2)

        @self.mcp.tool(
            name="ft_eval.export",
            description="Export domain knowledge evaluation results as JSONL or JSON",
        )
        async def ft_eval_export(
            results: List[Dict],
            output_path: str,
            format: ResultExportFormat = "jsonl",
        ) -> str:
            from model_evaluator_pipeline.models.ft_evaluator import FTEvalResult
            parsed = [FTEvalResult(**r) for r in results]
            result = await self.ft_evaluator.export_results(
                results=parsed,
                output_path=output_path,
                format=format,
            )
            return json.dumps(result, indent=2)

    # ------------------------------------------------------------------ #
    # Dataset tools
    # ------------------------------------------------------------------ #
    def _register_dataset_tools(self):
        @self.mcp.tool(
            name="dataset.save",
            description=(
                "Save data_points to a file (JSONL, JSON, or Parquet). Use after "
                "generate.from_document, clean.dataset, normalize.dataset, or "
                "evaluate.filter_by_quality to persist results to disk. "
                "Returns dataset_id (derived from filename) and file_path."
            ),
        )
        async def dataset_save(
            data_points: List[Dict],
            output_path: str,
            format: DatasetSaveFormat = "jsonl",
        ) -> str:
            return json.dumps(
                await self.dataset_service.save(data_points, output_path, format),
                indent=2,
            )

        @self.mcp.tool(
            name="dataset.load",
            description=(
                "Load a dataset from file and return data_points. "
                "Auto-detects format from extension (.jsonl, .json, .parquet, .csv). "
                "Pipe results into clean.dataset, evaluate.dataset, or finetune.train."
            ),
        )
        async def dataset_load(file_path: str) -> str:
            return json.dumps(
                await self.dataset_service.load(file_path), indent=2,
            )

        @self.mcp.tool(
            name="dataset.preview",
            description="Show first N rows from a dataset file without loading everything.",
        )
        async def dataset_preview(file_path: str, n: int = 5) -> str:
            return json.dumps(
                await self.dataset_service.preview(file_path, n), indent=2,
            )

        @self.mcp.tool(
            name="dataset.info",
            description=(
                "Get metadata about a dataset file: row count, columns, "
                "detected technique (sft/dpo/grpo/kto), file size."
            ),
        )
        async def dataset_info(file_path: str) -> str:
            return json.dumps(
                await self.dataset_service.info(file_path), indent=2,
            )

        @self.mcp.tool(
            name="dataset.delete",
            description="Delete a dataset file from disk.",
        )
        async def dataset_delete(file_path: str) -> str:
            return json.dumps(
                await self.dataset_service.delete(file_path), indent=2,
            )

        @self.mcp.tool(
            name="dataset.split",
            description=(
                "Split a dataset file into train/val/test files with configurable "
                "ratios and random seed. Returns paths to the three split files."
            ),
        )
        async def dataset_split(
            file_path: str,
            output_dir: str,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            seed: int = 42,
        ) -> str:
            return json.dumps(
                await self.dataset_service.split(
                    file_path, output_dir,
                    train_ratio=train_ratio, val_ratio=val_ratio,
                    test_ratio=test_ratio, seed=seed,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="dataset.merge",
            description=(
                "Merge multiple dataset files into one. "
                "Optionally deduplicate by a key (default: instruction)."
            ),
        )
        async def dataset_merge(
            file_paths: List[str],
            output_path: str,
            deduplicate: bool = False,
            dedup_key: str = "instruction",
        ) -> str:
            return json.dumps(
                await self.dataset_service.merge(
                    file_paths, output_path,
                    deduplicate=deduplicate, dedup_key=dedup_key,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="dataset.list",
            description=(
                "List all dataset files in the data directory. Returns metadata "
                "(id, path, format, row_count, columns, technique) for each file."
            ),
        )
        async def dataset_list(
            data_dir: str = "data",
        ) -> str:
            import os
            from pathlib import Path as _Path

            persisted = await self._persistence.list_datasets()
            datasets_by_path = {
                str(item.get("file_path")): item
                for item in persisted
                if item.get("file_path")
            }
            root = _Path(data_dir)
            if not root.exists():
                root = _Path(os.getcwd()) / data_dir
            if not root.exists():
                return json.dumps(
                    {
                        "success": True,
                        "datasets": list(datasets_by_path.values()),
                        "count": len(datasets_by_path),
                    },
                    indent=2,
                )

            supported = (".jsonl", ".json", ".csv", ".parquet")
            datasets = list(datasets_by_path.values())
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix.lower() in supported:
                    file_path = str(f.resolve())
                    if file_path in datasets_by_path:
                        continue
                    meta = await self.dataset_service.info(file_path)
                    if meta.get("success"):
                        datasets.append(meta["metadata"])

            return json.dumps({
                "success": True,
                "datasets": datasets,
                "count": len(datasets),
            }, indent=2)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def _wrap_tools_with_diagnostics(self) -> None:
        """Wrap all registered async tool functions with timing + emit_tool_call.

        Patches self.mcp._tools in-place after registration so no individual
        handler needs to be modified. No-op if DiagnosticWriter is not initialized.
        """
        from shared.diagnostics import emit_tool_call, sanitize

        def _make_wrapper(name: str, fn):
            async def _wrapper(**kwargs):
                # Generate a per-call trace_id (gateway is a separate process)
                from shared.diagnostics import trace_id_var
                trace_id_var.set(str(uuid.uuid4()))

                t0 = time.perf_counter()
                try:
                    result = await fn(**kwargs)
                    latency = round(time.perf_counter() - t0, 4)
                    preview = str(result)[:500] if result else ""
                    await emit_tool_call(
                        tool_name=name,
                        arguments=sanitize(kwargs),
                        result_preview=preview,
                        latency_s=latency,
                        success=True,
                    )
                    return result
                except Exception as exc:
                    latency = round(time.perf_counter() - t0, 4)
                    await emit_tool_call(
                        tool_name=name,
                        arguments=sanitize(kwargs),
                        result_preview="",
                        latency_s=latency,
                        success=False,
                        error=str(exc),
                    )
                    raise

            return _wrapper

        for tool_name in list(self.mcp._tools.keys()):
            tool_info = self.mcp._tools[tool_name]
            original_func = tool_info["func"]
            if inspect.iscoroutinefunction(original_func):
                tool_info["func"] = _make_wrapper(tool_name, original_func)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self, transport=None):
        # Set a gateway-scoped session_id before the event loop starts so
        # it propagates (via context copy) to all asyncio tasks.
        from shared.diagnostics import init_diagnostics, session_id_var
        gw_session = f"gw-{str(uuid.uuid4())[:8]}"
        session_id_var.set(gw_session)
        init_diagnostics(log_root="logs")
        try:
            asyncio.run(self._persistence.ensure_ready())
        except Exception:
            pass
        self.mcp.run(transport)


# Backwards-compatible aliases
AgentYGateway = TunaGateway
TranscendenceGateway = TunaGateway
