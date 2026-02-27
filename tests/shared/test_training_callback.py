"""Unit tests for ProgressCallback (HuggingFace TrainerCallback).

Tests verify that progress metrics are extracted from trainer state
and forwarded to the update function.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from shared.training_callback import ProgressCallback


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_state(
    global_step: int = 0,
    max_steps: int = 100,
    epoch: float = 0.0,
    log_history: Optional[List] = None,
):
    state = MagicMock()
    state.global_step = global_step
    state.max_steps = max_steps
    state.epoch = epoch
    state.log_history = log_history or []
    return state


def _make_args(num_train_epochs: int = 3):
    args = MagicMock()
    args.num_train_epochs = num_train_epochs
    return args


def _make_control():
    control = MagicMock()
    control.should_training_stop = False
    return control


class TestProgressCallback:
    def test_on_train_begin_sets_max_steps(self):
        updates: List[Dict[str, Any]] = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-1",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tempfile.mkdtemp(),
        )

        state = _make_state(max_steps=200)
        args = _make_args(num_train_epochs=5)
        control = _make_control()

        cb.on_train_begin(args, state, control)

        assert len(updates) == 1
        assert updates[0]["max_steps"] == 200
        assert updates[0]["max_epochs"] == 5
        assert updates[0]["current_step"] == 0
        assert updates[0]["percent_complete"] == 0.0

    def test_on_log_extracts_loss(self):
        updates: List[Dict[str, Any]] = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-2",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tempfile.mkdtemp(),
        )

        state = _make_state(global_step=10)
        args = _make_args()
        control = _make_control()
        logs = {"loss": 2.345, "learning_rate": 0.0002}

        cb.on_log(args, state, control, logs=logs)

        assert len(updates) == 1
        assert updates[0]["loss"] == 2.345
        assert updates[0]["learning_rate"] == 0.0002

    def test_on_log_extracts_eval_loss(self):
        updates: List[Dict[str, Any]] = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-3",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tempfile.mkdtemp(),
        )

        logs = {"eval_loss": 1.8, "eval_runtime": 5.2}
        cb.on_log(_make_args(), _make_state(), _make_control(), logs=logs)

        assert updates[0]["eval_loss"] == 1.8

    def test_on_log_extracts_grad_norm(self):
        updates: List[Dict[str, Any]] = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-4",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tempfile.mkdtemp(),
        )

        logs = {"loss": 1.0, "grad_norm": 0.85}
        cb.on_log(_make_args(), _make_state(), _make_control(), logs=logs)

        assert updates[0]["grad_norm"] == 0.85

    def test_on_log_none_logs_ignored(self):
        updates: List[Dict[str, Any]] = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-5",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tempfile.mkdtemp(),
        )

        cb.on_log(_make_args(), _make_state(), _make_control(), logs=None)
        assert len(updates) == 0

    def test_on_step_end_computes_percent_and_eta(self):
        updates: List[Dict[str, Any]] = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-6",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tempfile.mkdtemp(),
        )

        # Simulate train_begin to set start time
        cb._start_time = time.monotonic() - 10  # 10 seconds ago

        state = _make_state(global_step=50, max_steps=100, epoch=1.5)
        args = _make_args(num_train_epochs=3)
        control = _make_control()

        cb.on_step_end(args, state, control)

        assert len(updates) == 1
        assert updates[0]["current_step"] == 50
        assert updates[0]["percent_complete"] == 50.0
        assert updates[0]["current_epoch"] == 1.5
        # ETA should be approximately 10 seconds (50 steps in 10s, 50 remaining)
        assert updates[0]["eta_seconds"] > 0

    def test_on_step_end_checks_cancel_event(self):
        updates = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cancel = threading.Event()
        cancel.set()  # Pre-set cancellation

        cb = ProgressCallback(
            job_id="test-7",
            update_fn=capture_update,
            cancel_event=cancel,
            output_dir=tempfile.mkdtemp(),
        )
        cb._start_time = time.monotonic()

        control = _make_control()
        cb.on_step_end(_make_args(), _make_state(global_step=5), control)

        assert control.should_training_stop is True

    def test_on_step_end_does_not_stop_without_cancel(self):
        updates = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-8",
            update_fn=capture_update,
            cancel_event=threading.Event(),  # Not set
            output_dir=tempfile.mkdtemp(),
        )
        cb._start_time = time.monotonic()

        control = _make_control()
        cb.on_step_end(_make_args(), _make_state(global_step=5), control)

        assert control.should_training_stop is False

    def test_log_history_capped_at_50(self):
        updates = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        cb = ProgressCallback(
            job_id="test-9",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tempfile.mkdtemp(),
        )

        # Add 60 log entries
        for i in range(60):
            cb.on_log(
                _make_args(),
                _make_state(global_step=i),
                _make_control(),
                logs={"loss": float(i)},
            )

        assert len(cb._log_history) == 50
        # Should keep the most recent 50
        assert cb._log_history[0]["loss"] == 10.0
        assert cb._log_history[-1]["loss"] == 59.0

    def test_snapshot_written_periodically(self):
        updates = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        tmpdir = tempfile.mkdtemp()
        cb = ProgressCallback(
            job_id="test-10",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tmpdir,
            snapshot_interval=5,
        )
        cb._start_time = time.monotonic()

        # Step 5 should trigger snapshot
        cb.on_step_end(
            _make_args(),
            _make_state(global_step=5, max_steps=100),
            _make_control(),
        )

        snapshot_file = os.path.join(tmpdir, ".training_progress.jsonl")
        assert os.path.exists(snapshot_file)
        with open(snapshot_file) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["current_step"] == 5

    def test_on_train_end_writes_final_snapshot(self):
        updates = []

        def capture_update(job_id, **kwargs):
            updates.append(kwargs)

        tmpdir = tempfile.mkdtemp()
        cb = ProgressCallback(
            job_id="test-11",
            update_fn=capture_update,
            cancel_event=threading.Event(),
            output_dir=tmpdir,
        )
        cb._start_time = time.monotonic()

        state = _make_state(global_step=100, max_steps=100, epoch=3.0)
        cb.on_train_end(_make_args(), state, _make_control())

        snapshot_file = os.path.join(tmpdir, ".training_progress.jsonl")
        assert os.path.exists(snapshot_file)
        with open(snapshot_file) as f:
            data = json.loads(f.readlines()[-1])
        assert data["current_step"] == 100
        assert data.get("final") is True
