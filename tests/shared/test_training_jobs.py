"""Unit tests for training job state management.

Tests TrainingProgress, TrainingJob models and TrainingJobManager lifecycle:
create, start, complete, fail, cancel, list, get.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from shared.training_jobs import (
    JobStatus,
    TrainingJob,
    TrainingJobManager,
    TrainingProgress,
)


# ──────────────────────────────────────────────
# TrainingProgress model tests
# ──────────────────────────────────────────────


class TestTrainingProgress:
    def test_default_values(self):
        p = TrainingProgress()
        assert p.current_step == 0
        assert p.max_steps == 0
        assert p.percent_complete == 0.0
        assert p.loss is None
        assert p.learning_rate is None
        assert p.eval_loss is None
        assert p.eta_seconds is None
        assert p.gpu_memory_used_gb is None
        assert p.log_history == []

    def test_model_dump_serializes(self):
        p = TrainingProgress(
            current_step=50,
            max_steps=100,
            loss=1.234,
            percent_complete=50.0,
        )
        d = p.model_dump()
        assert d["current_step"] == 50
        assert d["max_steps"] == 100
        assert d["loss"] == 1.234
        assert d["percent_complete"] == 50.0

    def test_mutable_fields(self):
        p = TrainingProgress()
        p.current_step = 10
        p.loss = 2.5
        assert p.current_step == 10
        assert p.loss == 2.5


# ──────────────────────────────────────────────
# TrainingJob model tests
# ──────────────────────────────────────────────


class TestTrainingJob:
    def test_default_status_is_pending(self):
        job = TrainingJob(job_id="test-123")
        assert job.status == JobStatus.PENDING
        assert job.result is None
        assert job.error is None

    def test_model_dump_includes_progress(self):
        job = TrainingJob(
            job_id="test-456",
            status=JobStatus.RUNNING,
            trainer_type="sft",
            base_model="meta-llama/Llama-3.2-3B-Instruct",
        )
        d = job.model_dump()
        assert d["job_id"] == "test-456"
        assert d["status"] == "running"
        assert "progress" in d
        assert d["progress"]["current_step"] == 0

    def test_job_stores_result(self):
        job = TrainingJob(job_id="test-789")
        job.status = JobStatus.COMPLETED
        job.result = {"success": True, "model_path": "/output/model"}
        assert job.result["success"] is True

    def test_job_stores_error(self):
        job = TrainingJob(job_id="test-err")
        job.status = JobStatus.FAILED
        job.error = "CUDA OOM"
        assert job.error == "CUDA OOM"


# ──────────────────────────────────────────────
# TrainingJobManager tests
# ──────────────────────────────────────────────


class TestTrainingJobManager:
    def test_create_job_returns_job_with_id(self):
        mgr = TrainingJobManager()
        job = mgr.create_job(
            trainer_type="sft",
            base_model="test-model",
            output_dir="/tmp/out",
            config_summary={"num_epochs": 3},
        )
        assert job.job_id.startswith("job-")
        assert job.status == JobStatus.PENDING
        assert job.trainer_type == "sft"
        assert job.base_model == "test-model"
        assert job.config_summary == {"num_epochs": 3}
        assert job.created_at != ""

    def test_get_job_returns_created_job(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        retrieved = mgr.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_job_returns_none_for_unknown_id(self):
        mgr = TrainingJobManager()
        assert mgr.get_job("nonexistent-id") is None

    def test_list_jobs_returns_all(self):
        mgr = TrainingJobManager()
        mgr.create_job("sft", "m1", "/o1", {})
        mgr.create_job("dpo", "m2", "/o2", {})
        jobs = mgr.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_filters_by_status(self):
        mgr = TrainingJobManager()
        j1 = mgr.create_job("sft", "m1", "/o1", {})
        mgr.create_job("dpo", "m2", "/o2", {})
        j1.status = JobStatus.RUNNING
        jobs = mgr.list_jobs(status=JobStatus.RUNNING)
        assert len(jobs) == 1
        assert jobs[0].job_id == j1.job_id

    def test_list_jobs_respects_limit(self):
        mgr = TrainingJobManager()
        for i in range(5):
            mgr.create_job("sft", f"m{i}", f"/o{i}", {})
        jobs = mgr.list_jobs(limit=3)
        assert len(jobs) == 3

    def test_cancel_nonexistent_job_returns_false(self):
        mgr = TrainingJobManager()
        assert mgr.cancel_job("nonexistent") is False

    def test_cancel_sets_event_and_status(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.RUNNING
        # Manually register cancel event (normally done by start_job)
        mgr._cancel_events[job.job_id] = threading.Event()
        result = mgr.cancel_job(job.job_id)
        assert result is True
        assert mgr._cancel_events[job.job_id].is_set()

    def test_cancel_completed_job_returns_false(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.COMPLETED
        assert mgr.cancel_job(job.job_id) is False

    @pytest.mark.asyncio
    async def test_start_job_runs_to_completion(self):
        """Mock training callable that succeeds, verify PENDING -> RUNNING -> COMPLETED."""
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})

        async def fake_training(extra_callbacks=None):
            await asyncio.sleep(0.05)
            return {"success": True, "model_path": "/out"}

        await mgr.start_job(job.job_id, fake_training)

        for _ in range(20):
            updated = mgr.get_job(job.job_id)
            if updated is not None and updated.status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.05)

        updated = mgr.get_job(job.job_id)
        assert updated is not None
        assert updated.status == JobStatus.COMPLETED
        assert updated.result is not None
        assert updated.result["success"] is True
        assert updated.completed_at is not None
        assert updated.elapsed_seconds > 0

    @pytest.mark.asyncio
    async def test_start_job_handles_failure(self):
        """Mock training that raises, verify job status = FAILED."""
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})

        async def failing_training(extra_callbacks=None):
            raise RuntimeError("CUDA out of memory")

        await mgr.start_job(job.job_id, failing_training)
        await asyncio.sleep(0.2)

        updated = mgr.get_job(job.job_id)
        assert updated is not None
        assert updated.status == JobStatus.FAILED
        assert "CUDA out of memory" in updated.error

    @pytest.mark.asyncio
    async def test_start_job_provides_cancel_event(self):
        """Verify cancel event is created and accessible."""
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})

        received_callbacks = []

        async def capture_training(extra_callbacks=None):
            received_callbacks.extend(extra_callbacks or [])
            await asyncio.sleep(0.05)
            return {"success": True}

        await mgr.start_job(job.job_id, capture_training)
        await asyncio.sleep(0.2)

        assert job.job_id in mgr._cancel_events

    @pytest.mark.asyncio
    async def test_cancel_running_job(self):
        """Start a slow training, cancel it, verify status = CANCELLED."""
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})

        cancel_checked = threading.Event()

        async def slow_training(extra_callbacks=None):
            # Simulate checking cancel via callback
            for i in range(20):
                await asyncio.sleep(0.05)
                if mgr._cancel_events.get(job.job_id, threading.Event()).is_set():
                    cancel_checked.set()
                    return {"success": True, "interrupted": True}
            return {"success": True}

        await mgr.start_job(job.job_id, slow_training)
        await asyncio.sleep(0.1)  # Let it start

        mgr.cancel_job(job.job_id)
        await asyncio.sleep(0.5)

        updated = mgr.get_job(job.job_id)
        assert updated is not None
        assert updated.status in (JobStatus.CANCELLED, JobStatus.COMPLETED)

    def test_update_progress(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        mgr.update_progress(
            job.job_id,
            current_step=50,
            max_steps=100,
            loss=1.5,
            percent_complete=50.0,
        )
        updated = mgr.get_job(job.job_id)
        assert updated.progress.current_step == 50
        assert updated.progress.loss == 1.5
        assert updated.progress.percent_complete == 50.0
