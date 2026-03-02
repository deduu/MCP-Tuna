"""Integration tests for training monitoring MCP tools.

Tests the job_status, list_jobs, cancel_job tools and the
async training tool wrappers via a mocked gateway.
"""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

from shared.training_jobs import JobStatus, TrainingJobManager


# ──────────────────────────────────────────────
# TrainingJobManager direct integration tests
# ──────────────────────────────────────────────


class TestJobStatusIntegration:
    def test_job_status_returns_progress(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {"num_epochs": 3})
        job.status = JobStatus.RUNNING
        mgr.update_progress(
            job.job_id,
            current_step=63,
            max_steps=100,
            loss=1.234,
            percent_complete=63.0,
            eta_seconds=802.3,
            gpu_memory_used_gb=7.2,
            gpu_memory_total_gb=24.0,
        )

        retrieved = mgr.get_job(job.job_id)
        assert retrieved is not None

        data = retrieved.model_dump()
        assert data["status"] == "running"
        assert data["progress"]["current_step"] == 63
        assert data["progress"]["max_steps"] == 100
        assert data["progress"]["loss"] == 1.234
        assert data["progress"]["percent_complete"] == 63.0
        assert data["progress"]["eta_seconds"] == 802.3
        assert data["progress"]["gpu_memory_used_gb"] == 7.2

    def test_job_status_unknown_returns_none(self):
        mgr = TrainingJobManager()
        assert mgr.get_job("nonexistent") is None

    def test_completed_job_has_result(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.COMPLETED
        job.result = {"success": True, "model_path": "/out"}
        job.completed_at = "2026-02-27T10:15:00Z"
        job.elapsed_seconds = 900.0

        data = job.model_dump()
        assert data["status"] == "completed"
        assert data["result"]["success"] is True
        assert data["elapsed_seconds"] == 900.0


class TestListJobsIntegration:
    def test_list_all_jobs(self):
        mgr = TrainingJobManager()
        mgr.create_job("sft", "m1", "/o1", {})
        mgr.create_job("dpo", "m2", "/o2", {})
        mgr.create_job("grpo", "m3", "/o3", {})

        jobs = mgr.list_jobs()
        assert len(jobs) == 3
        types = [j.trainer_type for j in jobs]
        assert "sft" in types
        assert "dpo" in types

    def test_list_jobs_by_status(self):
        mgr = TrainingJobManager()
        j1 = mgr.create_job("sft", "m1", "/o1", {})
        j2 = mgr.create_job("dpo", "m2", "/o2", {})
        j1.status = JobStatus.RUNNING
        j2.status = JobStatus.COMPLETED

        running = mgr.list_jobs(status=JobStatus.RUNNING)
        assert len(running) == 1
        assert running[0].trainer_type == "sft"

        completed = mgr.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].trainer_type == "dpo"


class TestCancelJobIntegration:
    def test_cancel_running_job(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.RUNNING
        mgr._cancel_events[job.job_id] = threading.Event()

        assert mgr.cancel_job(job.job_id) is True
        assert mgr._cancel_events[job.job_id].is_set()

    def test_cancel_nonrunning_fails(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        # Still PENDING
        assert mgr.cancel_job(job.job_id) is False


class TestJobSerializationForMCP:
    """Verify that job.model_dump() produces JSON-serializable output
    suitable for MCP tool responses."""

    def test_full_job_is_json_serializable(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "meta-llama/Llama-3.2-3B", "/output/model", {
            "num_epochs": 3, "lora_r": 8,
        })
        job.status = JobStatus.RUNNING
        mgr.update_progress(
            job.job_id,
            current_step=50,
            max_steps=100,
            loss=1.5,
            learning_rate=0.0002,
            percent_complete=50.0,
            eta_seconds=450.0,
        )

        data = job.model_dump()
        # Must not raise
        serialized = json.dumps({"success": True, **data}, indent=2)
        parsed = json.loads(serialized)

        assert parsed["success"] is True
        assert parsed["job_id"] == job.job_id
        assert parsed["progress"]["current_step"] == 50
        assert parsed["progress"]["loss"] == 1.5

    def test_list_jobs_json_serializable(self):
        mgr = TrainingJobManager()
        mgr.create_job("sft", "m1", "/o1", {})
        mgr.create_job("dpo", "m2", "/o2", {})

        jobs = mgr.list_jobs()
        data = {
            "success": True,
            "count": len(jobs),
            "jobs": [j.model_dump() for j in jobs],
        }
        # Must not raise
        serialized = json.dumps(data, indent=2)
        parsed = json.loads(serialized)
        assert parsed["count"] == 2


class TestAsyncJobLifecycle:
    """End-to-end async lifecycle: create -> start -> progress -> complete."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {"num_epochs": 3})
        assert job.status == JobStatus.PENDING

        async def mock_training(extra_callbacks=None):
            # Simulate progress updates
            mgr.update_progress(job.job_id, current_step=50, max_steps=100, percent_complete=50.0)
            await asyncio.sleep(0.05)
            mgr.update_progress(job.job_id, current_step=100, max_steps=100, percent_complete=100.0)
            return {"success": True, "model_path": "/out"}

        await mgr.start_job(job.job_id, mock_training)
        await asyncio.sleep(0.2)

        final = mgr.get_job(job.job_id)
        assert final.status == JobStatus.COMPLETED
        assert final.result["success"] is True
        assert final.progress.percent_complete == 100.0

    @pytest.mark.asyncio
    async def test_failed_job_lifecycle(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})

        async def failing_training(extra_callbacks=None):
            raise RuntimeError("GPU out of memory")

        await mgr.start_job(job.job_id, failing_training)
        await asyncio.sleep(0.2)

        final = mgr.get_job(job.job_id)
        assert final.status == JobStatus.FAILED
        assert "GPU out of memory" in final.error
