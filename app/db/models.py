from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobRunRecord(Base):
    __tablename__ = "job_runs"
    __table_args__ = (
        UniqueConstraint("namespace", "job_id", name="uq_job_runs_namespace_job_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    namespace: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    trainer_type: Mapped[str] = mapped_column(String(64), default="")
    base_model: Mapped[str] = mapped_column(Text, default="")
    output_dir: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    elapsed_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    progress: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class DeploymentRecord(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    type: Mapped[str] = mapped_column(String(32), default="mcp")
    modality: Mapped[str] = mapped_column(String(32), default="text")
    transport: Mapped[str] = mapped_column(String(32), default="http")
    host: Mapped[str] = mapped_column(String(255), default="127.0.0.1")
    port: Mapped[int] = mapped_column(Integer, default=0)
    endpoint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    api_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    routes: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_path: Mapped[str] = mapped_column(Text, default="")
    adapter_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    deployment_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    modality: Mapped[str] = mapped_column(String(32), default="text")
    endpoint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    adapter_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    messages: Mapped[list["ConversationMessageRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessageRecord.sequence",
    )


class ConversationMessageRecord(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(32))
    content_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    conversation: Mapped[ConversationRecord] = relationship(back_populates="messages")


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), index=True)
    file_path: Mapped[str] = mapped_column(Text, unique=True)
    format: Mapped[str] = mapped_column(String(32))
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    columns: Mapped[list[str]] = mapped_column(JSON, default=list)
    technique: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    modified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    object_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    object_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bucket: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    object_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    object_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )
