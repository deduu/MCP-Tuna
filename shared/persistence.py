from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Select, desc, select

from app.core.config import settings
from app.db.models import (
    ArtifactRecord,
    ConversationMessageRecord,
    ConversationRecord,
    DatasetRecord,
    DeploymentRecord,
    JobRunRecord,
)
from app.db.session import session_manager

logger = logging.getLogger(__name__)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _normalize_job_status(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        value = enum_value

    if isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith("JobStatus."):
            return normalized.split(".", 1)[1].lower()
        return normalized

    if value is None:
        return "pending"

    return str(value)


class PersistenceService:
    """Best-effort PostgreSQL persistence for runtime metadata."""

    def __init__(self) -> None:
        self._enabled = settings.persistence.enabled
        self._initialized = False
        self._init_failed = False
        self._init_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._init_failed

    async def ensure_ready(self) -> bool:
        if not self._enabled:
            return False
        if self._initialized:
            return True
        if self._init_failed:
            return False

        async with self._init_lock:
            if self._initialized:
                return True
            try:
                await session_manager.create_tables()
                self._initialized = True
                logger.info("Persistence tables are ready")
            except Exception as exc:
                self._init_failed = True
                logger.warning("Persistence unavailable: %s", exc)
                return False
        return True

    async def _with_session(self, fn):
        if not await self.ensure_ready():
            return None
        try:
            async with session_manager.session() as session:
                return await fn(session)
        except Exception as exc:
            logger.warning("Persistence operation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    @staticmethod
    def _job_to_dict(record: JobRunRecord) -> Dict[str, Any]:
        return {
            "job_id": record.job_id,
            "status": _normalize_job_status(record.status),
            "trainer_type": record.trainer_type,
            "base_model": record.base_model,
            "output_dir": record.output_dir,
            "created_at": _to_iso(record.created_at) or "",
            "started_at": _to_iso(record.started_at),
            "completed_at": _to_iso(record.completed_at),
            "elapsed_seconds": record.elapsed_seconds,
            "progress": record.progress or {},
            "result": record.result,
            "error": record.error,
            "config_summary": record.config_summary or {},
        }

    async def upsert_job(self, namespace: str, job: Dict[str, Any]) -> bool:
        async def _op(session):
            stmt = select(JobRunRecord).where(
                JobRunRecord.namespace == namespace,
                JobRunRecord.job_id == job["job_id"],
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                record = JobRunRecord(namespace=namespace, job_id=job["job_id"])
                session.add(record)

            record.status = _normalize_job_status(job.get("status", record.status or "pending"))
            record.trainer_type = str(job.get("trainer_type") or "")
            record.base_model = str(job.get("base_model") or "")
            record.output_dir = str(job.get("output_dir") or "")
            record.created_at = _parse_dt(job.get("created_at")) or record.created_at
            record.started_at = _parse_dt(job.get("started_at"))
            record.completed_at = _parse_dt(job.get("completed_at"))
            record.elapsed_seconds = float(job.get("elapsed_seconds") or 0.0)
            record.progress = job.get("progress") or {}
            record.result = job.get("result")
            record.error = job.get("error")
            record.config_summary = job.get("config_summary") or {}
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    async def get_job(self, namespace: str, job_id: str) -> Optional[Dict[str, Any]]:
        async def _op(session):
            stmt = select(JobRunRecord).where(
                JobRunRecord.namespace == namespace,
                JobRunRecord.job_id == job_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            return self._job_to_dict(record) if record is not None else None

        result = await self._with_session(_op)
        return result if isinstance(result, dict) else None

    async def list_jobs(
        self,
        namespace: str,
        *,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        async def _op(session):
            stmt: Select[tuple[JobRunRecord]] = select(JobRunRecord).where(
                JobRunRecord.namespace == namespace,
            )
            if status:
                stmt = stmt.where(JobRunRecord.status == status)
            stmt = stmt.order_by(desc(JobRunRecord.updated_at)).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._job_to_dict(row) for row in rows]

        result = await self._with_session(_op)
        return result if isinstance(result, list) else []

    async def delete_job(self, namespace: str, job_id: str) -> bool:
        async def _op(session):
            stmt = select(JobRunRecord).where(
                JobRunRecord.namespace == namespace,
                JobRunRecord.job_id == job_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                return False
            await session.delete(record)
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    # ------------------------------------------------------------------
    # Deployments
    # ------------------------------------------------------------------

    @staticmethod
    def _deployment_to_dict(record: DeploymentRecord) -> Dict[str, Any]:
        metadata = record.metadata_json or {}
        return {
            "deployment_id": record.deployment_id,
            "name": metadata.get("name"),
            "status": record.status,
            "type": record.type,
            "modality": record.modality,
            "transport": record.transport,
            "host": record.host,
            "port": record.port,
            "endpoint": record.endpoint,
            "api_path": record.api_path,
            "routes": record.routes or [],
            "model_path": record.model_path,
            "adapter_path": record.adapter_path,
            "metadata": metadata,
            "created_at": _to_iso(record.created_at),
            "updated_at": _to_iso(record.updated_at),
            "stopped_at": _to_iso(record.stopped_at),
        }

    async def upsert_deployment(self, deployment: Dict[str, Any]) -> bool:
        async def _op(session):
            stmt = select(DeploymentRecord).where(
                DeploymentRecord.deployment_id == deployment["deployment_id"],
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                record = DeploymentRecord(deployment_id=deployment["deployment_id"])
                session.add(record)

            record.status = str(deployment.get("status") or "running")
            record.type = str(deployment.get("type") or "mcp")
            record.modality = str(deployment.get("modality") or "text")
            record.transport = str(deployment.get("transport") or "http")
            record.host = str(deployment.get("host") or "127.0.0.1")
            record.port = int(deployment.get("port") or 0)
            record.endpoint = deployment.get("endpoint")
            record.api_path = deployment.get("api_path")
            record.routes = deployment.get("routes") or []
            record.model_path = str(deployment.get("model_path") or "")
            record.adapter_path = deployment.get("adapter_path")
            record.metadata_json = deployment.get("metadata") or {}
            if record.status == "stopped":
                record.stopped_at = utc_now()
            elif deployment.get("stopped_at"):
                record.stopped_at = _parse_dt(deployment.get("stopped_at"))
            else:
                record.stopped_at = None
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    async def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        async def _op(session):
            stmt = select(DeploymentRecord).where(
                DeploymentRecord.deployment_id == deployment_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            return self._deployment_to_dict(record) if record is not None else None

        result = await self._with_session(_op)
        return result if isinstance(result, dict) else None

    async def list_deployments(self, *, include_stopped: bool = True) -> List[Dict[str, Any]]:
        async def _op(session):
            stmt: Select[tuple[DeploymentRecord]] = select(DeploymentRecord)
            if not include_stopped:
                stmt = stmt.where(DeploymentRecord.status != "stopped")
            stmt = stmt.order_by(desc(DeploymentRecord.updated_at))
            rows = (await session.execute(stmt)).scalars().all()
            return [self._deployment_to_dict(row) for row in rows]

        result = await self._with_session(_op)
        return result if isinstance(result, list) else []

    async def delete_deployment(self, deployment_id: str) -> bool:
        async def _op(session):
            stmt = select(DeploymentRecord).where(
                DeploymentRecord.deployment_id == deployment_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                return False
            await session.delete(record)
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    @staticmethod
    def _conversation_summary_to_dict(record: ConversationRecord) -> Dict[str, Any]:
        metadata = record.metadata_json or {}
        return {
            "conversation_id": record.conversation_id,
            "deployment_id": record.deployment_id,
            "modality": record.modality,
            "endpoint": record.endpoint,
            "model_path": record.model_path,
            "adapter_path": record.adapter_path,
            "message_count": record.message_count,
            "metadata": metadata,
            "title": metadata.get("title"),
            "created_at": _to_iso(record.created_at),
            "updated_at": _to_iso(record.updated_at),
        }

    @staticmethod
    def _conversation_to_dict(record: ConversationRecord) -> Dict[str, Any]:
        return {
            **PersistenceService._conversation_summary_to_dict(record),
            "system_prompt": record.system_prompt,
            "messages": [
                {
                    "sequence": message.sequence,
                    "role": message.role,
                    "content": message.content_json
                    if message.content_json is not None
                    else message.content_text,
                }
                for message in sorted(record.messages, key=lambda item: item.sequence)
            ],
        }

    async def upsert_conversation(self, conversation: Dict[str, Any]) -> bool:
        async def _op(session):
            stmt = select(ConversationRecord).where(
                ConversationRecord.conversation_id == conversation["conversation_id"],
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                record = ConversationRecord(conversation_id=conversation["conversation_id"])
                session.add(record)

            record.deployment_id = conversation.get("deployment_id")
            record.modality = str(conversation.get("modality") or "text")
            record.endpoint = conversation.get("endpoint")
            record.model_path = conversation.get("model_path")
            record.adapter_path = conversation.get("adapter_path")
            record.system_prompt = conversation.get("system_prompt")
            record.metadata_json = conversation.get("metadata") or {}
            if conversation.get("message_count") is not None:
                record.message_count = int(conversation["message_count"])
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    async def append_conversation_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: Any,
    ) -> bool:
        async def _op(session):
            stmt = select(ConversationRecord).where(
                ConversationRecord.conversation_id == conversation_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                return False

            next_sequence = record.message_count + 1
            message = ConversationMessageRecord(
                conversation_id=record.id,
                sequence=next_sequence,
                role=role,
                content_text=content if isinstance(content, str) else None,
                content_json=None if isinstance(content, str) else content,
            )
            session.add(message)
            record.message_count = next_sequence
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        async def _op(session):
            stmt = select(ConversationRecord).where(
                ConversationRecord.conversation_id == conversation_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                return None
            await session.refresh(record, attribute_names=["messages"])
            return self._conversation_to_dict(record)

        result = await self._with_session(_op)
        return result if isinstance(result, dict) else None

    async def set_conversation_title(self, conversation_id: str, title: str) -> bool:
        async def _op(session):
            stmt = select(ConversationRecord).where(
                ConversationRecord.conversation_id == conversation_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                return False
            metadata = dict(record.metadata_json or {})
            metadata["title"] = title
            record.metadata_json = metadata
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    async def delete_conversation(self, conversation_id: str) -> bool:
        async def _op(session):
            stmt = select(ConversationRecord).where(
                ConversationRecord.conversation_id == conversation_id,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                return False
            await session.delete(record)
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    async def list_conversations(
        self,
        *,
        deployment_id: Optional[str] = None,
        modality: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        async def _op(session):
            stmt: Select[tuple[ConversationRecord]] = select(ConversationRecord)
            if deployment_id:
                stmt = stmt.where(ConversationRecord.deployment_id == deployment_id)
            if modality:
                stmt = stmt.where(ConversationRecord.modality == modality)
            stmt = stmt.order_by(desc(ConversationRecord.updated_at)).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._conversation_summary_to_dict(row) for row in rows]

        result = await self._with_session(_op)
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    @staticmethod
    def _dataset_to_dict(record: DatasetRecord) -> Dict[str, Any]:
        metadata = {
            "dataset_id": record.dataset_id,
            "file_path": record.file_path,
            "format": record.format,
            "row_count": record.row_count,
            "columns": record.columns or [],
            "technique": record.technique,
            "size_bytes": record.size_bytes,
            "modified_at": _to_iso(record.modified_at),
        }
        if record.object_key:
            metadata["object_key"] = record.object_key
        if record.object_url:
            metadata["object_url"] = record.object_url
        return metadata

    async def upsert_dataset(self, metadata: Dict[str, Any]) -> bool:
        async def _op(session):
            stmt = select(DatasetRecord).where(
                DatasetRecord.file_path == metadata["file_path"],
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                record = DatasetRecord(file_path=metadata["file_path"])
                session.add(record)

            record.dataset_id = str(metadata.get("dataset_id") or "")
            record.format = str(metadata.get("format") or "")
            record.row_count = int(metadata.get("row_count") or 0)
            record.columns = metadata.get("columns") or []
            record.technique = metadata.get("technique")
            record.size_bytes = int(metadata.get("size_bytes") or 0)
            record.modified_at = _parse_dt(metadata.get("modified_at"))
            record.object_key = metadata.get("object_key")
            record.object_url = metadata.get("object_url")
            record.metadata_json = metadata.get("metadata") or {}
            record.deleted_at = None
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    async def get_dataset(self, file_path: str) -> Optional[Dict[str, Any]]:
        async def _op(session):
            stmt = select(DatasetRecord).where(
                DatasetRecord.file_path == file_path,
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            return self._dataset_to_dict(record) if record is not None else None

        result = await self._with_session(_op)
        return result if isinstance(result, dict) else None

    async def list_datasets(self) -> List[Dict[str, Any]]:
        async def _op(session):
            stmt: Select[tuple[DatasetRecord]] = (
                select(DatasetRecord)
                .where(DatasetRecord.deleted_at.is_(None))
                .order_by(desc(DatasetRecord.updated_at))
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [self._dataset_to_dict(row) for row in rows]

        result = await self._with_session(_op)
        return result if isinstance(result, list) else []

    async def mark_dataset_deleted(self, file_path: str) -> bool:
        async def _op(session):
            stmt = select(DatasetRecord).where(DatasetRecord.file_path == file_path)
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                return False
            record.deleted_at = utc_now()
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def upsert_artifact(self, artifact: Dict[str, Any]) -> bool:
        async def _op(session):
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_key == artifact["artifact_key"],
            )
            record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                record = ArtifactRecord(artifact_key=artifact["artifact_key"])
                session.add(record)

            record.kind = str(artifact.get("kind") or "artifact")
            record.local_path = artifact.get("local_path")
            record.bucket = artifact.get("bucket")
            record.object_key = artifact.get("object_key")
            record.object_url = artifact.get("object_url")
            record.metadata_json = artifact.get("metadata") or {}
            await session.commit()
            return True

        result = await self._with_session(_op)
        return bool(result)


_persistence_service = PersistenceService()


def get_persistence_service() -> PersistenceService:
    return _persistence_service


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
