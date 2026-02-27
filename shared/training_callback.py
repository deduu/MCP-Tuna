"""HuggingFace TrainerCallback for real-time progress reporting.

Imported lazily inside training methods so that ``transformers`` is NOT
loaded at gateway startup.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import — resolved inside the training thread.
try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover – only in unit tests w/o transformers
    TrainerCallback = object  # type: ignore[assignment,misc]


class ProgressCallback(TrainerCallback):  # type: ignore[misc]
    """Reports training progress to :class:`TrainingJobManager`.

    Instantiated inside the training thread and attached to the HuggingFace
    ``Trainer`` via the ``callbacks`` argument.

    Parameters
    ----------
    job_id:
        Unique training job identifier.
    update_fn:
        Callable that accepts ``(job_id, **progress_kwargs)`` and writes
        the values into the in-memory ``TrainingJob``.
    cancel_event:
        ``threading.Event`` — when set, the next ``on_step_end`` sets
        ``control.should_training_stop = True``.
    output_dir:
        Directory for periodic JSONL progress snapshots.
    snapshot_interval:
        Write a snapshot every *N* training steps (default 10).
    """

    def __init__(
        self,
        job_id: str,
        update_fn: Callable[..., Any],
        cancel_event: threading.Event,
        output_dir: str,
        snapshot_interval: int = 10,
    ):
        self.job_id = job_id
        self._update_fn = update_fn
        self._cancel_event = cancel_event
        self._output_dir = output_dir
        self._snapshot_interval = snapshot_interval
        self._start_time: float = 0.0
        self._log_history: List[Dict[str, Any]] = []

    # ── callbacks ──

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._start_time = time.monotonic()
        self._update_fn(
            self.job_id,
            current_step=0,
            max_steps=state.max_steps,
            max_epochs=int(args.num_train_epochs),
            percent_complete=0.0,
        )

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if logs is None:
            return

        update: Dict[str, Any] = {}
        if "loss" in logs:
            update["loss"] = logs["loss"]
        if "learning_rate" in logs:
            update["learning_rate"] = logs["learning_rate"]
        if "eval_loss" in logs:
            update["eval_loss"] = logs["eval_loss"]
        if "grad_norm" in logs:
            update["grad_norm"] = logs["grad_norm"]

        # Track log history (capped)
        self._log_history.append({**logs, "step": state.global_step})
        if len(self._log_history) > 50:
            self._log_history = self._log_history[-50:]

        if update:
            update["log_history"] = list(self._log_history)
            self._update_fn(self.job_id, **update)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        elapsed = time.monotonic() - self._start_time
        steps_done = state.global_step
        max_steps = max(state.max_steps, 1)
        steps_remaining = max_steps - steps_done
        avg_step_time = elapsed / max(steps_done, 1)
        eta = round(avg_step_time * steps_remaining, 1)
        percent = round(steps_done / max_steps * 100, 1)

        gpu_used, gpu_total = self._get_gpu_memory()

        self._update_fn(
            self.job_id,
            current_step=steps_done,
            max_steps=state.max_steps,
            current_epoch=round(state.epoch or 0.0, 2),
            max_epochs=int(args.num_train_epochs),
            percent_complete=percent,
            eta_seconds=eta,
            gpu_memory_used_gb=gpu_used,
            gpu_memory_total_gb=gpu_total,
        )

        # Check cancellation
        if self._cancel_event.is_set():
            control.should_training_stop = True

        # Periodic JSONL snapshot
        if steps_done > 0 and steps_done % self._snapshot_interval == 0:
            self._write_snapshot(steps_done)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write_snapshot(state.global_step, final=True)

    # ── helpers ──

    @staticmethod
    def _get_gpu_memory() -> tuple[Optional[float], Optional[float]]:
        try:
            import torch

            if torch.cuda.is_available():
                used = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
                total = round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 2)
                return used, total
        except Exception:
            pass
        return None, None

    def _write_snapshot(self, step: int, final: bool = False) -> None:
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            path = os.path.join(self._output_dir, ".training_progress.jsonl")
            entry: Dict[str, Any] = {
                "job_id": self.job_id,
                "current_step": step,
                "timestamp": time.time(),
            }
            if final:
                entry["final"] = True
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.debug("Failed to write progress snapshot", exc_info=True)
