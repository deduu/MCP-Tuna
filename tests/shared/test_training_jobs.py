"""Unit tests for training job state management.

Tests TrainingProgress, TrainingJob models and TrainingJobManager lifecycle:
create, start, complete, fail, cancel, list, get.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

import shared.training_jobs as training_jobs_module
from shared.ownership import (
    reset_current_ownership_context,
    set_current_ownership_context,
)
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

    def test_model_validate_accepts_legacy_prefixed_status(self):
        job = TrainingJob.model_validate({
            "job_id": "test-legacy",
            "status": "JobStatus.COMPLETED",
        })
        assert job.status == JobStatus.COMPLETED

    def test_model_dump_json_uses_plain_status_value(self):
        job = TrainingJob(job_id="test-json", status=JobStatus.RUNNING)
        payload = job.model_dump(mode="json")
        assert payload["status"] == "running"

    def test_ownership_defaults_to_local_workspace_context(self):
        job = TrainingJob(job_id="test-owner")
        assert job.ownership.workspace_id

    def test_model_validate_accepts_ownership_dict(self):
        job = TrainingJob.model_validate(
            {
                "job_id": "test-owner-dict",
                "ownership": {"workspace_id": "alpha-ws", "user_id": "user-1"},
            }
        )
        assert job.ownership.workspace_id == "alpha-ws"
        assert job.ownership.user_id == "user-1"


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
            ownership={"workspace_id": "alpha-ws", "user_id": "user-1"},
        )
        assert job.job_id.startswith("job-")
        assert job.status == JobStatus.PENDING
        assert job.trainer_type == "sft"
        assert job.base_model == "test-model"
        assert job.config_summary == {"num_epochs": 3}
        assert job.ownership.workspace_id == "alpha-ws"
        assert job.ownership.user_id == "user-1"
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

    def test_create_job_uses_current_context_when_ownership_omitted(self):
        mgr = TrainingJobManager()
        token = set_current_ownership_context(
            {"workspace_id": "ctx-ws", "user_id": "ctx-user"}
        )
        try:
            job = mgr.create_job("sft", "model", "/out", {})
        finally:
            reset_current_ownership_context(token)

        assert job.ownership.workspace_id == "ctx-ws"
        assert job.ownership.user_id == "ctx-user"

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

    def test_list_jobs_filters_by_ownership(self):
        mgr = TrainingJobManager()
        mgr.create_job(
            "sft",
            "m1",
            "/o1",
            {},
            ownership={"workspace_id": "alpha-ws", "user_id": "user-1"},
        )
        mgr.create_job(
            "dpo",
            "m2",
            "/o2",
            {},
            ownership={"workspace_id": "beta-ws", "user_id": "user-2"},
        )

        jobs = mgr.list_jobs(ownership={"workspace_id": "alpha-ws", "user_id": "user-1"})

        assert len(jobs) == 1
        assert jobs[0].trainer_type == "sft"

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

    def test_cancel_pending_queued_job_marks_cancelled(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.PENDING
        mgr._cancel_events[job.job_id] = threading.Event()

        result = mgr.cancel_job(job.job_id)

        assert result is True
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None
        assert job.progress.current_stage == "cancelled"

    def test_cancel_orphaned_running_job_marks_cancelled(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.RUNNING

        result = mgr.cancel_job(job.job_id)

        assert result is True
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_delete_finished_job_removes_record(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.FAILED

        result = await mgr.adelete_job(job.job_id)

        assert result is True
        assert mgr.get_job(job.job_id) is None

    @pytest.mark.asyncio
    async def test_delete_active_job_is_rejected(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.RUNNING
        mgr._cancel_events[job.job_id] = threading.Event()

        result = await mgr.adelete_job(job.job_id)

        assert result is False
        assert mgr.get_job(job.job_id) is job

    @pytest.mark.asyncio
    async def test_acancel_job_marks_pending_record_cancelled_without_live_worker(self):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})

        result = await mgr.acancel_job(job.job_id)

        assert result is True
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_acancel_job_marks_pending_queued_job_cancelled_even_with_live_task(self):
        mgr = TrainingJobManager(max_concurrent=1)
        first = mgr.create_job("sft", "model", "/out/one", {})
        second = mgr.create_job("sft", "model", "/out/two", {})
        release_first = threading.Event()
        first_started = threading.Event()

        async def slow_training(extra_callbacks=None):
            first_started.set()
            while not release_first.is_set():
                await asyncio.sleep(0.01)
            return {"success": True}

        async def fast_training(extra_callbacks=None):
            return {"success": True}

        await mgr.start_job(first.job_id, slow_training)
        while not first_started.is_set():
            await asyncio.sleep(0.01)

        await mgr.start_job(second.job_id, fast_training)
        await asyncio.sleep(0.05)

        result = await mgr.acancel_job(second.job_id)
        updated = mgr.get_job(second.job_id)

        assert result is True
        assert updated is not None
        assert updated.status == JobStatus.CANCELLED
        assert updated.progress.current_stage == "cancelled"

        release_first.set()
        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_acancel_job_marks_persisted_orphan_running_job_cancelled(self):
        mgr = TrainingJobManager()
        payload = TrainingJob(
            job_id="job-persisted",
            status=JobStatus.RUNNING,
            trainer_type="sft",
            base_model="model",
            output_dir="/out",
            created_at="2026-03-20T00:00:00+00:00",
        ).model_dump(mode="json")

        async def fake_get_job(namespace, job_id, ownership=None):
            assert namespace == "training"
            assert job_id == "job-persisted"
            assert ownership is None
            return payload

        mgr._persistence.get_job = fake_get_job  # type: ignore[method-assign]

        result = await mgr.acancel_job("job-persisted")
        updated = mgr.get_job("job-persisted")

        assert result is True
        assert updated is not None
        assert updated.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_aget_job_marks_persisted_orphan_running_job_failed(self):
        mgr = TrainingJobManager()
        payload = TrainingJob(
            job_id="job-orphaned",
            status=JobStatus.RUNNING,
            trainer_type="sft",
            base_model="model",
            output_dir="/out",
            created_at="2026-04-09T09:12:56+00:00",
            started_at="2026-04-09T09:12:57+00:00",
        ).model_dump(mode="json")

        mgr._persistence.get_job = AsyncMock(return_value=payload)  # type: ignore[method-assign]
        mgr._persistence.upsert_job = AsyncMock(return_value=True)  # type: ignore[method-assign]

        job = await mgr.aget_job("job-orphaned")

        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert job.progress.current_stage == "failed"
        assert "backend restarted" in (job.error or "").lower()
        mgr._persistence.upsert_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_alist_jobs_excludes_orphaned_running_job_from_active_filter(self):
        mgr = TrainingJobManager()
        payload = TrainingJob(
            job_id="job-orphaned",
            status=JobStatus.RUNNING,
            trainer_type="sft",
            base_model="model",
            output_dir="/out",
            created_at="2026-04-09T09:12:56+00:00",
            started_at="2026-04-09T09:12:57+00:00",
        ).model_dump(mode="json")

        mgr._persistence.list_jobs = AsyncMock(return_value=[payload])  # type: ignore[method-assign]
        mgr._persistence.upsert_job = AsyncMock(return_value=True)  # type: ignore[method-assign]

        running = await mgr.alist_jobs(status=JobStatus.RUNNING)
        all_jobs = await mgr.alist_jobs()

        assert running == []
        assert len(all_jobs) == 1
        assert all_jobs[0].status == JobStatus.FAILED
        mgr._persistence.upsert_job.assert_awaited()

    @pytest.mark.asyncio
    async def test_adelete_job_rejects_persisted_record_for_other_owner(self):
        mgr = TrainingJobManager()
        payload = TrainingJob(
            job_id="job-owned",
            status=JobStatus.COMPLETED,
            trainer_type="sft",
            base_model="model",
            output_dir="/out",
            created_at="2026-03-20T00:00:00+00:00",
            ownership={"workspace_id": "beta-ws", "user_id": "user-2"},
        ).model_dump(mode="json")

        mgr._persistence.get_job = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._persistence.delete_job = AsyncMock(return_value=False)  # type: ignore[method-assign]

        result = await mgr.adelete_job(
            "job-owned",
            ownership={"workspace_id": "alpha-ws", "user_id": "user-1"},
        )

        assert result is False
        mgr._persistence.delete_job.assert_not_awaited()

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
    async def test_start_job_sets_pending_queue_state_before_worker_runs(self):
        mgr = TrainingJobManager(max_concurrent=1)
        job = mgr.create_job("sft", "model", "/out", {})
        gate = threading.Event()

        async def blocked_training(extra_callbacks=None):
            while not gate.is_set():
                await asyncio.sleep(0.01)
            return {"success": True}

        await mgr.start_job(job.job_id, blocked_training)

        queued = mgr.get_job(job.job_id)
        assert queued is not None
        assert queued.status == JobStatus.PENDING
        assert queued.progress.current_stage == "queued"
        assert queued.progress.status_message == "Queued for execution"

        gate.set()
        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_second_job_stays_pending_while_first_worker_is_busy(self):
        mgr = TrainingJobManager(max_concurrent=1)
        first = mgr.create_job("sft", "model", "/out/one", {})
        second = mgr.create_job("sft", "model", "/out/two", {})
        release_first = threading.Event()
        first_started = threading.Event()

        async def slow_training(extra_callbacks=None):
            first_started.set()
            while not release_first.is_set():
                await asyncio.sleep(0.01)
            return {"success": True}

        async def fast_training(extra_callbacks=None):
            return {"success": True}

        await mgr.start_job(first.job_id, slow_training)
        while not first_started.is_set():
            await asyncio.sleep(0.01)

        await mgr.start_job(second.job_id, fast_training)
        await asyncio.sleep(0.05)

        updated_first = mgr.get_job(first.job_id)
        updated_second = mgr.get_job(second.job_id)
        assert updated_first is not None
        assert updated_second is not None
        assert updated_first.status == JobStatus.RUNNING
        assert updated_first.progress.current_stage == "startup"
        assert updated_second.status == JobStatus.PENDING
        assert updated_second.progress.current_stage == "queued"

        release_first.set()
        await asyncio.sleep(0.2)

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

    @pytest.mark.asyncio
    async def test_alist_jobs_returns_local_jobs_when_persistence_is_slow(self, monkeypatch):
        mgr = TrainingJobManager()
        job = mgr.create_job("sft", "model", "/out", {})
        job.status = JobStatus.RUNNING

        async def slow_list_jobs(*args, **kwargs):
            await asyncio.sleep(0.05)
            return []

        monkeypatch.setattr(
            training_jobs_module,
            "_PERSISTED_JOB_LIST_TIMEOUT_WITH_LOCAL_JOBS_S",
            0.01,
        )
        mgr._persistence.list_jobs = slow_list_jobs  # type: ignore[method-assign]

        jobs = await mgr.alist_jobs()

        assert len(jobs) == 1
        assert jobs[0].job_id == job.job_id
