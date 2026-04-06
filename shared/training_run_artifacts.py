from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _readable_path(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _infer_dataset_kind(columns: Sequence[str]) -> str:
    column_set = set(columns)
    if {"prompt", "chosen", "rejected"}.issubset(column_set):
        return "dpo"
    if {"prompt", "responses", "rewards"}.issubset(column_set):
        return "grpo"
    if {"prompt", "completion", "label"}.issubset(column_set):
        return "kto"
    if "messages" in column_set:
        return "vlm_sft"
    if {"system", "user", "assistant"}.issubset(column_set):
        return "chat_triplet_sft"
    if {"prompt", "response"}.issubset(column_set):
        return "prompt_response_sft"
    if {"instruction", "output"}.issubset(column_set):
        return "instruction_output_sft"
    return "unknown"


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _dataset_length(dataset: Any) -> int | None:
    try:
        return int(len(dataset))
    except Exception:
        return None


def _column_names(dataset: Any, sample_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    raw_columns = getattr(dataset, "column_names", None)
    if isinstance(raw_columns, Sequence) and not isinstance(raw_columns, (str, bytes)):
        return [str(column) for column in raw_columns]

    columns: list[str] = []
    seen: set[str] = set()
    for row in sample_rows:
        for key in row:
            text = str(key)
            if text not in seen:
                seen.add(text)
                columns.append(text)
    return columns


def _sample_rows(dataset: Any, limit: int = 128) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(dataset, list):
        iterable = dataset[:limit]
        for row in iterable:
            if isinstance(row, Mapping):
                rows.append(dict(row))
        return rows

    length = _dataset_length(dataset)
    if length is None:
        return rows

    for index in range(min(length, limit)):
        try:
            row = dataset[index]
        except Exception:
            break
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _mean_text_length(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    lengths = [
        len(str(value).strip())
        for value in (row.get(field) for row in rows)
        if isinstance(value, str) and value.strip()
    ]
    if not lengths:
        return None
    return round(mean(lengths), 2)


def _duplicate_ratio(values: Iterable[str]) -> float | None:
    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    unique = len(set(cleaned))
    return round(1.0 - (unique / len(cleaned)), 4)


def _summarize_sft_like_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_field: str,
    response_field: str,
) -> dict[str, Any]:
    prompts = [
        str(row.get(prompt_field) or "").strip()
        for row in rows
        if str(row.get(prompt_field) or "").strip()
    ]
    return {
        "avg_prompt_chars": _mean_text_length(rows, prompt_field),
        "avg_response_chars": _mean_text_length(rows, response_field),
        "duplicate_prompt_ratio": _duplicate_ratio(prompts),
    }


def _summarize_dpo_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompts = [
        str(row.get("prompt") or "").strip()
        for row in rows
        if str(row.get("prompt") or "").strip()
    ]
    chosen = [
        str(row.get("chosen") or "").strip()
        for row in rows
        if str(row.get("chosen") or "").strip()
    ]
    rejected = [
        str(row.get("rejected") or "").strip()
        for row in rows
        if str(row.get("rejected") or "").strip()
    ]
    return {
        "avg_prompt_chars": _mean_text_length(rows, "prompt"),
        "avg_chosen_chars": _mean_text_length(rows, "chosen"),
        "avg_rejected_chars": _mean_text_length(rows, "rejected"),
        "duplicate_prompt_ratio": _duplicate_ratio(prompts),
        "chosen_duplicate_ratio": _duplicate_ratio(chosen),
        "rejected_duplicate_ratio": _duplicate_ratio(rejected),
    }


def _summarize_grpo_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    response_counts: list[int] = []
    response_lengths: list[int] = []
    reward_means: list[float] = []
    reward_stds: list[float] = []

    for row in rows:
        responses = row.get("responses")
        rewards = row.get("rewards")
        if isinstance(responses, Sequence) and not isinstance(responses, (str, bytes)):
            response_counts.append(len(responses))
            response_lengths.extend(
                len(str(item).strip())
                for item in responses
                if str(item).strip()
            )
        if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes)):
            reward_values = [
                value for value in (_safe_float(item) for item in rewards)
                if value is not None
            ]
            if reward_values:
                reward_means.append(mean(reward_values))
                reward_stds.append(pstdev(reward_values))

    zero_variance = sum(1 for value in reward_stds if value == 0.0)
    return {
        "avg_prompt_chars": _mean_text_length(rows, "prompt"),
        "avg_responses_per_row": _rounded(mean(response_counts)) if response_counts else None,
        "avg_response_chars": round(mean(response_lengths), 2) if response_lengths else None,
        "avg_reward_mean": _rounded(mean(reward_means)) if reward_means else None,
        "avg_reward_std": _rounded(mean(reward_stds)) if reward_stds else None,
        "zero_reward_variance_ratio": (
            _rounded(zero_variance / len(reward_stds)) if reward_stds else None
        ),
    }


def _summarize_kto_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true_count = 0
    false_count = 0
    for row in rows:
        label = row.get("label")
        if label is True:
            true_count += 1
        elif label is False:
            false_count += 1
    total = true_count + false_count
    return {
        "avg_prompt_chars": _mean_text_length(rows, "prompt"),
        "avg_completion_chars": _mean_text_length(rows, "completion"),
        "desirable_count": true_count,
        "undesirable_count": false_count,
        "desirable_ratio": _rounded(true_count / total) if total else None,
    }


def _summarize_vlm_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    message_counts: list[int] = []
    multimodal_rows = 0
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            continue
        message_counts.append(len(messages))
        has_non_text = False
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            if any(
                isinstance(block, Mapping) and str(block.get("type") or "").strip() not in {"", "text"}
                for block in content
            ):
                has_non_text = True
                break
        if has_non_text:
            multimodal_rows += 1
    return {
        "avg_messages_per_row": _rounded(mean(message_counts)) if message_counts else None,
        "multimodal_row_ratio": (
            _rounded(multimodal_rows / len(message_counts)) if message_counts else None
        ),
    }


def summarize_training_dataset(
    dataset: Any,
    *,
    dataset_path: Optional[str] = None,
    sample_limit: int = 128,
) -> dict[str, Any]:
    sample_rows = _sample_rows(dataset, limit=sample_limit)
    columns = _column_names(dataset, sample_rows)
    dataset_kind = _infer_dataset_kind(columns)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_kind": dataset_kind,
        "dataset_path": _readable_path(dataset_path),
        "num_examples": _dataset_length(dataset),
        "columns": columns,
        "sample_size": len(sample_rows),
    }

    if dataset_path:
        path = Path(dataset_path)
        if path.exists() and path.is_file():
            summary["source_file"] = _hash_file(path)

    if dataset_kind == "dpo":
        summary["statistics"] = _summarize_dpo_rows(sample_rows)
    elif dataset_kind == "grpo":
        summary["statistics"] = _summarize_grpo_rows(sample_rows)
    elif dataset_kind == "kto":
        summary["statistics"] = _summarize_kto_rows(sample_rows)
    elif dataset_kind == "vlm_sft":
        summary["statistics"] = _summarize_vlm_rows(sample_rows)
    elif dataset_kind == "chat_triplet_sft":
        summary["statistics"] = _summarize_sft_like_rows(
            sample_rows,
            prompt_field="user",
            response_field="assistant",
        )
    elif dataset_kind == "instruction_output_sft":
        summary["statistics"] = _summarize_sft_like_rows(
            sample_rows,
            prompt_field="instruction",
            response_field="output",
        )
    elif dataset_kind == "prompt_response_sft":
        summary["statistics"] = _summarize_sft_like_rows(
            sample_rows,
            prompt_field="prompt",
            response_field="response",
        )
    else:
        summary["statistics"] = {}

    return summary


class TrainingRunArtifacts:
    """Structured per-run artifacts for finetuning runs."""

    def __init__(
        self,
        *,
        output_dir: str,
        trainer: str,
    ) -> None:
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.trainer = trainer
        self.created_at = _utc_now_iso()
        self.artifacts: dict[str, str] = {}
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "trainer": self.trainer,
            "created_at": self.created_at,
            "output_dir": str(self.output_path),
            "invocation": {
                "run_source": None,
                "job_id": None,
                "note": None,
            },
            "model": {
                "base_model": None,
                "adapter_path": None,
            },
            "training_config": {},
            "dataset": {},
        }
        self._status: dict[str, Any] = {
            "schema_version": 1,
            "trainer": self.trainer,
            "state": "pending",
            "success": None,
            "interrupted": False,
            "created_at": self.created_at,
            "updated_at": self.created_at,
            "completed_at": None,
            "output_dir": str(self.output_path),
            "model_path": None,
            "error": None,
            "artifacts": {},
        }

    def _artifact_path(self, filename: str) -> Path:
        return self.output_path / filename

    def write_json_artifact(
        self,
        key: str,
        filename: str,
        payload: Mapping[str, Any],
    ) -> str:
        path = self._artifact_path(filename)
        path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.artifacts[key] = str(path)
        return str(path)

    def write_markdown_artifact(
        self,
        key: str,
        filename: str,
        content: str,
    ) -> str:
        path = self._artifact_path(filename)
        path.write_text(content, encoding="utf-8")
        self.artifacts[key] = str(path)
        return str(path)

    def start(
        self,
        *,
        base_model: Optional[str],
        adapter_path: Optional[str],
        dataset: Any,
        dataset_path: Optional[str],
        training_config: Mapping[str, Any],
        run_source: Optional[str] = None,
        job_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        dataset_diagnostics = summarize_training_dataset(
            dataset,
            dataset_path=dataset_path,
        )
        self._manifest = {
            "schema_version": 1,
            "trainer": self.trainer,
            "created_at": self.created_at,
            "output_dir": str(self.output_path),
            "invocation": {
                "run_source": _readable_path(run_source),
                "job_id": _readable_path(job_id),
                "note": _readable_path(note),
            },
            "model": {
                "base_model": _readable_path(base_model),
                "adapter_path": _readable_path(adapter_path),
            },
            "training_config": dict(training_config),
            "dataset": dataset_diagnostics,
        }
        self._status = {
            "schema_version": 1,
            "trainer": self.trainer,
            "state": "running",
            "success": None,
            "interrupted": False,
            "created_at": self.created_at,
            "updated_at": self.created_at,
            "completed_at": None,
            "output_dir": str(self.output_path),
            "model_path": None,
            "error": None,
            "artifacts": {},
        }
        self.write_json_artifact("dataset_diagnostics", "dataset_diagnostics.json", dataset_diagnostics)
        self.write_json_artifact("run_manifest", "run_manifest.json", self._manifest)
        self._write_status()
        self._write_summary()
        self._write_status()
        return dataset_diagnostics

    def complete(
        self,
        *,
        success: bool,
        interrupted: bool,
        model_path: Optional[str],
        error: Optional[str],
        training_time_seconds: Optional[float],
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, str]:
        self._status.update(
            {
                "state": (
                    "interrupted" if interrupted and success
                    else ("completed" if success else "failed")
                ),
                "success": success,
                "interrupted": interrupted,
                "updated_at": _utc_now_iso(),
                "completed_at": _utc_now_iso(),
                "model_path": _readable_path(model_path),
                "error": _readable_path(error),
                "training_time_seconds": (
                    round(float(training_time_seconds), 4)
                    if training_time_seconds is not None
                    else None
                ),
                "metrics": dict(metrics or {}),
            }
        )
        if not success and error:
            self.write_json_artifact(
                "failure",
                "failure.json",
                {
                    "trainer": self.trainer,
                    "error": error,
                    "output_dir": str(self.output_path),
                    "updated_at": self._status["updated_at"],
                },
            )
        self._write_status()
        self._write_summary()
        self._write_status()
        return dict(self.artifacts)

    def _write_status(self) -> None:
        status_path = self._artifact_path("run_status.json")
        artifacts = dict(self.artifacts)
        artifacts["run_status"] = str(status_path)
        self._status["artifacts"] = artifacts
        status_path.write_text(
            json.dumps(dict(self._status), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.artifacts["run_status"] = str(status_path)

    def _write_summary(self) -> None:
        dataset = self._manifest.get("dataset") or {}
        invocation = self._manifest.get("invocation") or {}
        lines = [
            "# Training Run Summary",
            "",
            f"- Trainer: {self.trainer}",
            f"- Status: {self._status.get('state', 'unknown')}",
            f"- Output dir: {self.output_path}",
            f"- Base model: {self._manifest.get('model', {}).get('base_model') or 'unknown'}",
            f"- Adapter init: {self._manifest.get('model', {}).get('adapter_path') or 'none'}",
            f"- Dataset path: {dataset.get('dataset_path') or 'not recorded'}",
            f"- Dataset kind: {dataset.get('dataset_kind') or 'unknown'}",
            f"- Num examples: {dataset.get('num_examples') if dataset.get('num_examples') is not None else 'unknown'}",
            f"- Run source: {invocation.get('run_source') or 'not recorded'}",
            f"- Job id: {invocation.get('job_id') or 'not recorded'}",
            f"- Created at: {self.created_at}",
            f"- Completed at: {self._status.get('completed_at') or 'in progress'}",
        ]
        training_time = self._status.get("training_time_seconds")
        if training_time is not None:
            lines.append(f"- Training time (s): {training_time}")
        error = self._status.get("error")
        if error:
            lines.append(f"- Error: {error}")
        metrics = self._status.get("metrics") or {}
        if metrics:
            lines.extend(["", "## Metrics"])
            for key, value in metrics.items():
                lines.append(f"- {key}: {value}")
        if self.artifacts:
            lines.extend(["", "## Artifacts"])
            for key, value in sorted(self.artifacts.items()):
                lines.append(f"- {key}: {value}")
        self.write_markdown_artifact("summary", "summary.md", "\n".join(lines) + "\n")
