from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from shared.training_jobs import JobStatus


class _FakeFinetuner:
    def __init__(self) -> None:
        self.config = SimpleNamespace(base_model="meta-llama/Llama-3.2-3B-Instruct")
        self.started = threading.Event()
        self.received_callbacks = None

    async def load_dataset_from_file(self, dataset_path: str, format: str) -> dict:
        return {
            "success": True,
            "dataset_object": [{"prompt": "q1", "response": "a1"}],
            "dataset_path": dataset_path,
            "format": format,
        }

    async def train_model(self, dataset, output_dir: str, extra_callbacks=None, **kwargs) -> dict:
        self.received_callbacks = list(extra_callbacks or [])
        self.started.set()
        while True:
            cancelled = any(
                getattr(callback, "_cancel_event", None) is not None
                and callback._cancel_event.is_set()
                for callback in self.received_callbacks
            )
            if cancelled:
                return {"success": False, "error": "cancelled", "output_dir": output_dir}
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_workflow_run_pipeline_async_cancels_custom_training_step(tmp_path: Path):
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    fake_finetuner = _FakeFinetuner()
    gateway._finetuning_svc = fake_finetuner

    run_pipeline_async = gateway.mcp._tools["workflow.run_pipeline_async"]["func"]
    cancel_job = gateway.mcp._tools["workflow.cancel_job"]["func"]

    workflow_output_dir = tmp_path / "workflow"
    steps = json.dumps(
        [
            {
                "tool": "finetune.train",
                "params": {
                    "dataset_path": str(tmp_path / "dataset.jsonl"),
                    "output_dir": str(tmp_path / "trained-model"),
                },
            }
        ]
    )

    start_payload = json.loads(
        await run_pipeline_async(steps=steps, output_dir=str(workflow_output_dir))
    )
    job_id = start_payload["job_id"]

    for _ in range(20):
        if await asyncio.to_thread(fake_finetuner.started.wait, 0.1):
            break
    assert fake_finetuner.started.is_set()
    assert fake_finetuner.received_callbacks

    cancel_payload = json.loads(await cancel_job(job_id=job_id))
    assert cancel_payload["success"] is True

    for _ in range(50):
        job = gateway.workflow_job_manager.get_job(job_id)
        if job is not None and job.status == JobStatus.CANCELLED:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("Workflow job did not transition to cancelled status")

    job = gateway.workflow_job_manager.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.CANCELLED
    assert job.progress.current_stage == "finetune.train"
    assert job.result is not None
    assert "cancelled" in job.result["error"].lower()


@pytest.mark.asyncio
async def test_workflow_delete_job_removes_finished_record(tmp_path: Path):
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    delete_job = gateway.mcp._tools["workflow.delete_job"]["func"]

    job = gateway.workflow_job_manager.create_job(
        trainer_type="workflow",
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=str(tmp_path / "workflow"),
        config_summary={"steps": ["finetune.train"]},
    )
    job.status = JobStatus.FAILED

    payload = json.loads(await delete_job(job_id=job.job_id))

    assert payload["success"] is True
    assert gateway.workflow_job_manager.get_job(job.job_id) is None
