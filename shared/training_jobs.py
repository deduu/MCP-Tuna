"""Training job state management for async training monitoring.

Provides TrainingJob / TrainingProgress (Pydantic v2 models), JobStatus enum,
and TrainingJobManager (singleton coordinator for background training jobs).
"""
from __future__ import annotations

import asyncio
import enum
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from shared.persistence import get_persistence_service

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Enums & data models
# ──────────────────────────────────────────────


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _normalize_job_status(value: Any) -> Any:
    if isinstance(value, JobStatus):
        return value

    if isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith("JobStatus."):
            normalized = normalized.split(".", 1)[1].lower()
        try:
            return JobStatus(normalized)
        except ValueError:
            return value

    return value


class TrainingProgress(BaseModel):
    """Real-time snapshot of training progress."""

    model_config = {"frozen": False}

    current_step: int = 0
    max_steps: int = 0
    current_epoch: float = 0.0
    max_epochs: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    eval_loss: Optional[float] = None
    grad_norm: Optional[float] = None
    eta_seconds: Optional[float] = None
    percent_complete: float = 0.0
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    last_updated: str = ""
    log_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_stage: Optional[str] = None
    status_message: Optional[str] = None
    stage_current: Optional[int] = None
    stage_total: Optional[int] = None
    stage_unit: Optional[str] = None


class TrainingJob(BaseModel):
    """Full state of a training job."""

    model_config = {"frozen": False}

    job_id: str
    status: JobStatus = JobStatus.PENDING
    trainer_type: str = "sft"
    base_model: str = ""
    output_dir: str = ""
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    progress: TrainingProgress = Field(default_factory=TrainingProgress)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    config_summary: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, value: Any) -> Any:
        return _normalize_job_status(value)


# ──────────────────────────────────────────────
# Job manager
# ──────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingJobManager:
    """Coordinates background training jobs.

    Manages lifecycle: create -> start (in thread) -> monitor -> complete/fail/cancel.
    Thread-safe via lock for dict mutations.
    """

    def __init__(self, max_concurrent: int = 1, namespace: str = "training"):
        self._jobs: Dict[str, TrainingJob] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="training"
        )
        self._lock = threading.Lock()
        self._namespace = namespace
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._persistence = get_persistence_service()

    # ── create ──

    def create_job(
        self,
        trainer_type: str,
        base_model: str,
        output_dir: str,
        config_summary: Dict[str, Any],
    ) -> TrainingJob:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        job = TrainingJob(
            job_id=job_id,
            trainer_type=trainer_type,
            base_model=base_model,
            output_dir=output_dir,
            created_at=_now_iso(),
            config_summary=config_summary,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._schedule_persist(job)
        return job

    # ── start ──

    async def start_job(
        self,
        job_id: str,
        training_coro: Callable,
    ) -> None:
        """Launch *training_coro* as a background asyncio task.

        *training_coro* must be an async callable that accepts
        ``extra_callbacks: Optional[List]`` and returns a result dict.
        """
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")

        cancel_event = threading.Event()
        self._loop = asyncio.get_running_loop()
        with self._lock:
            self._cancel_events[job_id] = cancel_event

        job.status = JobStatus.RUNNING
        job.started_at = _now_iso()
        self._schedule_persist(job)

        async def _run() -> None:
            start = time.monotonic()
            try:
                from shared.training_callback import ProgressCallback

                callback = ProgressCallback(
                    job_id=job_id,
                    update_fn=self.update_progress,
                    cancel_event=cancel_event,
                    output_dir=job.output_dir,
                )

                def _run_in_worker() -> Any:
                    return asyncio.run(training_coro(extra_callbacks=[callback]))

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(self._executor, _run_in_worker)

                elapsed = time.monotonic() - start
                if cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                else:
                    job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = _now_iso()
                job.elapsed_seconds = round(elapsed, 2)
                self._schedule_persist(job)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                if hasattr(exc, "result"):
                    job.result = getattr(exc, "result")
                job.completed_at = _now_iso()
                job.elapsed_seconds = round(time.monotonic() - start, 2)
                logger.exception("Training job %s failed", job_id)
                self._schedule_persist(job)

        task = asyncio.create_task(_run())
        with self._lock:
            self._tasks[job_id] = task

    # ── query ──

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 20,
    ) -> List[TrainingJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs[:limit]

    # ── cancel ──

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False

        task = self._tasks.get(job_id)
        event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
        elif task is None or task.done():
            # Persisted orphan record: no live worker remains to stop, so mark it cancelled.
            job.status = JobStatus.CANCELLED
            job.completed_at = job.completed_at or _now_iso()
            self._schedule_persist(job)
            return True
        else:
            return False
        self._schedule_persist(job)
        return True

    def get_cancel_event(self, job_id: str) -> Optional[threading.Event]:
        return self._cancel_events.get(job_id)

    async def acancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            persisted = await self._persistence.get_job(self._namespace, job_id)
            if persisted is None:
                return False
            job = TrainingJob.model_validate(persisted)
            with self._lock:
                self._jobs[job_id] = job

        if job.status not in (JobStatus.RUNNING, JobStatus.PENDING):
            return False

        event = self._cancel_events.get(job_id)
        task = self._tasks.get(job_id)
        if event is not None:
            event.set()
            self._schedule_persist(job)
            return True

        if task is not None and not task.done():
            return False

        # Persisted orphan record or queued job with no live worker: cancel in place.
        job.status = JobStatus.CANCELLED
        job.completed_at = job.completed_at or _now_iso()
        self._schedule_persist(job)
        return True

    async def adelete_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is not None and job.status in (JobStatus.RUNNING, JobStatus.PENDING):
            return False

        removed = False
        with self._lock:
            removed = self._jobs.pop(job_id, None) is not None or removed
            self._tasks.pop(job_id, None)
            self._cancel_events.pop(job_id, None)

        persisted_removed = await self._persistence.delete_job(self._namespace, job_id)
        return removed or persisted_removed

    # ── progress update ──

    def update_progress(self, job_id: str, **kwargs: Any) -> None:
        """Update specific progress fields on a job (thread-safe)."""
        job = self.get_job(job_id)
        if job is None:
            return
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(job.progress, key):
                    setattr(job.progress, key, value)
            job.progress.last_updated = _now_iso()
        self._schedule_persist(job)

    async def aget_job(self, job_id: str) -> Optional[TrainingJob]:
        job = self.get_job(job_id)
        if job is not None:
            return job

        persisted = await self._persistence.get_job(self._namespace, job_id)
        if persisted is None:
            return None
        return TrainingJob.model_validate(persisted)

    async def alist_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 20,
    ) -> List[TrainingJob]:
        jobs = self.list_jobs(status=status, limit=limit)
        persisted = await self._persistence.list_jobs(
            self._namespace,
            status=status.value if status is not None else None,
            limit=limit,
        )
        merged: Dict[str, TrainingJob] = {job.job_id: job for job in jobs}
        for item in persisted:
            job = TrainingJob.model_validate(item)
            merged.setdefault(job.job_id, job)

        return sorted(
            merged.values(),
            key=lambda job: job.created_at or "",
            reverse=True,
        )[:limit]

    def _schedule_persist(self, job: TrainingJob) -> None:
        payload = job.model_dump(mode="json")
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._persistence.upsert_job(self._namespace, payload),
                self._loop,
            )
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._persistence.upsert_job(self._namespace, payload))
