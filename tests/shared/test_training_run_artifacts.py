from __future__ import annotations

import json
from pathlib import Path

from shared.ownership import reset_current_ownership_context, set_current_ownership_context
from shared.training_run_artifacts import TrainingRunArtifacts


def test_complete_without_start_writes_valid_status(tmp_path):
    run_artifacts = TrainingRunArtifacts(output_dir=str(tmp_path), trainer="dpo")

    paths = run_artifacts.complete(
        success=False,
        interrupted=False,
        model_path=None,
        error="boom",
        training_time_seconds=1.25,
        metrics={},
    )

    status = json.loads(Path(paths["run_status"]).read_text(encoding="utf-8"))

    assert status["trainer"] == "dpo"
    assert status["state"] == "failed"
    assert status["error"] == "boom"
    assert status["ownership"]["workspace_id"]
    assert status["artifacts"]["run_status"] == paths["run_status"]
    assert status["artifacts"]["summary"] == paths["summary"]
    assert status["artifacts"]["failure"] == paths["failure"]


def test_complete_without_start_uses_current_ownership_context(tmp_path):
    token = set_current_ownership_context(
        {"workspace_id": "ctx-ws", "user_id": "ctx-user"}
    )
    try:
        run_artifacts = TrainingRunArtifacts(output_dir=str(tmp_path), trainer="dpo")
    finally:
        reset_current_ownership_context(token)

    paths = run_artifacts.complete(
        success=False,
        interrupted=False,
        model_path=None,
        error="boom",
        training_time_seconds=1.25,
        metrics={},
    )

    status = json.loads(Path(paths["run_status"]).read_text(encoding="utf-8"))

    assert status["ownership"] == {
        "workspace_id": "ctx-ws",
        "user_id": "ctx-user",
    }


def test_start_and_complete_failure_records_dataset_and_failure_artifacts(tmp_path):
    run_artifacts = TrainingRunArtifacts(output_dir=str(tmp_path), trainer="dpo")

    run_artifacts.start(
        base_model="base-model",
        adapter_path="adapter-path",
        dataset=[
            {
                "prompt": "Apa itu Salestify?",
                "chosen": "Salestify membantu tim sales mengelola chat WhatsApp.",
                "rejected": "Saya tidak tahu.",
            }
        ],
        dataset_path=None,
        training_config={"num_epochs": 2},
        run_source="finetune.train_dpo",
        job_id="job-123",
        note="test-run",
        ownership={"workspace_id": "alpha-ws", "user_id": "user-1"},
    )
    paths = run_artifacts.complete(
        success=False,
        interrupted=False,
        model_path=None,
        error="trainer crashed",
        training_time_seconds=2.5,
        metrics={"loss": 0.4},
    )

    status = json.loads(Path(paths["run_status"]).read_text(encoding="utf-8"))
    dataset_diagnostics = json.loads(
        Path(paths["dataset_diagnostics"]).read_text(encoding="utf-8")
    )
    manifest = json.loads(Path(paths["run_manifest"]).read_text(encoding="utf-8"))
    summary = Path(paths["summary"]).read_text(encoding="utf-8")

    assert dataset_diagnostics["dataset_kind"] == "dpo"
    assert manifest["ownership"]["workspace_id"] == "alpha-ws"
    assert manifest["ownership"]["user_id"] == "user-1"
    assert status["artifacts"]["failure"] == paths["failure"]
    assert status["ownership"]["workspace_id"] == "alpha-ws"
    assert "Dataset kind: dpo" in summary
    assert "Run source: finetune.train_dpo" in summary
    assert "Workspace: alpha-ws" in summary


def test_start_records_text_messages_dataset_kind(tmp_path):
    run_artifacts = TrainingRunArtifacts(output_dir=str(tmp_path), trainer="sft")

    run_artifacts.start(
        base_model="base-model",
        adapter_path=None,
        dataset=[
            {
                "messages": [
                    {"role": "user", "content": "Check the docs."},
                    {
                        "role": "assistant",
                        "content": "Searching.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": {"query": "docs"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "search_docs",
                        "content": "The docs recommend split-server configs.",
                    },
                    {
                        "role": "assistant",
                        "content": "Use split-server configs.",
                    },
                ]
            }
        ],
        dataset_path=None,
        training_config={"num_epochs": 1},
        run_source="finetune.train",
        ownership={"workspace_id": "alpha-ws", "user_id": "user-1"},
    )
    paths = run_artifacts.complete(
        success=True,
        interrupted=False,
        model_path=str(tmp_path / "model"),
        error=None,
        training_time_seconds=1.0,
        metrics={},
    )

    dataset_diagnostics = json.loads(
        Path(paths["dataset_diagnostics"]).read_text(encoding="utf-8")
    )

    assert dataset_diagnostics["dataset_kind"] == "messages_sft"
    assert dataset_diagnostics["statistics"]["avg_tool_calls_per_row"] == 1.0
