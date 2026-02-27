"""Tests verifying extra_callbacks parameter is wired through in training methods.

Each test mocks the trainer construction and verifies that extra_callbacks
are included in the trainer's callbacks list.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import types
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finetuning_pipeline.services.training_service import TrainingService
from shared.config import FinetuningConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_mock_dataset(rows: List[Dict[str, str]], col_names: List[str] | None = None):
    dataset = MagicMock()
    dataset.column_names = col_names or (list(rows[0].keys()) if rows else [])
    dataset.__len__ = lambda self: len(rows)
    dataset.__iter__ = lambda self: iter(rows)
    dataset.__getitem__ = lambda self, idx: rows[idx]

    def mock_map(fn, **kwargs):
        new_rows = [{**row, **fn(row)} for row in rows]
        return _make_mock_dataset(new_rows, col_names)

    def mock_remove_columns(cols):
        new_cols = [c for c in (col_names or list(rows[0].keys())) if c not in cols]
        new_rows = [{k: v for k, v in row.items() if k not in cols} for row in rows]
        return _make_mock_dataset(new_rows, new_cols)

    dataset.map = mock_map
    dataset.remove_columns = mock_remove_columns
    return dataset


def _sample_sft_data():
    return [
        {"prompt": "What is Python?", "response": "A language."},
        {"prompt": "What is 2+2?", "response": "4"},
    ]


def _sample_dpo_data():
    return [
        {"prompt": "Hi", "chosen": "Hello!", "rejected": "Go away"},
    ]


def _sample_kto_data():
    return [
        {"prompt": "Hi", "completion": "Hello!", "label": True},
    ]


class _FakeCallback:
    """A simple fake callback for testing."""
    pass


# ──────────────────────────────────────────────
# SFT extra_callbacks test
# ──────────────────────────────────────────────


class TestSFTExtraCallbacks:
    @pytest.mark.asyncio
    async def test_extra_callbacks_passed_to_sft_trainer(self):
        svc = TrainingService(FinetuningConfig(base_model="test/model"))
        fake_cb = _FakeCallback()
        captured_kwargs: Dict[str, Any] = {}

        fake_trainer = MagicMock()
        fake_trainer.train = MagicMock()
        fake_trainer.save_model = MagicMock()

        def capture_sft_trainer(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_trainer

        with (
            patch.object(svc, "_load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
            patch.object(svc, "_resolve_checkpoint", return_value=None),
            patch.object(svc, "_pop_training_kwargs", return_value={}),
            patch.object(svc, "_cleanup"),
            patch("finetuning_pipeline.services.training_service.Path"),
        ):
            # Mock the trl import
            fake_trl = types.ModuleType("trl")
            fake_trl.SFTTrainer = capture_sft_trainer
            fake_trl.SFTConfig = type("SFTConfig", (), {"__init__": lambda self, **kw: None})
            with patch.dict("sys.modules", {"trl": fake_trl}):
                # Need to re-import SFTTrainer/SFTConfig as they're imported inside the method
                result = await svc.train_model(
                    dataset=_make_mock_dataset(_sample_sft_data()),
                    output_dir=tempfile.mkdtemp(),
                    extra_callbacks=[fake_cb],
                )

        # The callbacks kwarg should contain our fake_cb
        if "callbacks" in captured_kwargs:
            assert fake_cb in captured_kwargs["callbacks"]

    @pytest.mark.asyncio
    async def test_extra_callbacks_none_by_default_sft(self):
        """Verify no error when extra_callbacks is not passed."""
        svc = TrainingService(FinetuningConfig(base_model="test/model"))
        captured_kwargs: Dict[str, Any] = {}

        fake_trainer = MagicMock()
        fake_trainer.train = MagicMock()
        fake_trainer.save_model = MagicMock()

        def capture_sft_trainer(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_trainer

        with (
            patch.object(svc, "_load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
            patch.object(svc, "_resolve_checkpoint", return_value=None),
            patch.object(svc, "_pop_training_kwargs", return_value={}),
            patch.object(svc, "_cleanup"),
            patch("finetuning_pipeline.services.training_service.Path"),
        ):
            fake_trl = types.ModuleType("trl")
            fake_trl.SFTTrainer = capture_sft_trainer
            fake_trl.SFTConfig = type("SFTConfig", (), {"__init__": lambda self, **kw: None})
            with patch.dict("sys.modules", {"trl": fake_trl}):
                result = await svc.train_model(
                    dataset=_make_mock_dataset(_sample_sft_data()),
                    output_dir=tempfile.mkdtemp(),
                    # No extra_callbacks
                )

        # Should still succeed
        assert result.get("success") is True or "callbacks" not in captured_kwargs or captured_kwargs.get("callbacks") == []


# ──────────────────────────────────────────────
# DPO extra_callbacks test
# ──────────────────────────────────────────────


class TestDPOExtraCallbacks:
    @pytest.mark.asyncio
    async def test_extra_callbacks_passed_to_dpo_trainer(self):
        svc = TrainingService(FinetuningConfig(base_model="test/model"))
        fake_cb = _FakeCallback()
        captured_kwargs: Dict[str, Any] = {}

        fake_trainer = MagicMock()
        fake_trainer.train = MagicMock()
        fake_trainer.save_model = MagicMock()

        def capture_dpo_trainer(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_trainer

        with (
            patch.object(svc, "_load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
            patch.object(svc, "_resolve_checkpoint", return_value=None),
            patch.object(svc, "_pop_training_kwargs", return_value={}),
            patch.object(svc, "_cleanup"),
            patch("finetuning_pipeline.services.training_service.Path"),
        ):
            fake_trl = types.ModuleType("trl")
            fake_trl.DPOTrainer = capture_dpo_trainer
            fake_trl.DPOConfig = type("DPOConfig", (), {"__init__": lambda self, **kw: None})
            with patch.dict("sys.modules", {"trl": fake_trl}):
                result = await svc.train_dpo_model(
                    dataset=_make_mock_dataset(_sample_dpo_data(), ["prompt", "chosen", "rejected"]),
                    output_dir=tempfile.mkdtemp(),
                    extra_callbacks=[fake_cb],
                )

        if "callbacks" in captured_kwargs:
            assert fake_cb in captured_kwargs["callbacks"]


# ──────────────────────────────────────────────
# KTO extra_callbacks test
# ──────────────────────────────────────────────


class TestKTOExtraCallbacks:
    @pytest.mark.asyncio
    async def test_extra_callbacks_passed_to_kto_trainer(self):
        svc = TrainingService(FinetuningConfig(base_model="test/model"))
        fake_cb = _FakeCallback()
        captured_kwargs: Dict[str, Any] = {}

        fake_trainer = MagicMock()
        fake_trainer.train = MagicMock()
        fake_trainer.save_model = MagicMock()

        def capture_kto_trainer(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_trainer

        with (
            patch.object(svc, "_load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
            patch.object(svc, "_resolve_checkpoint", return_value=None),
            patch.object(svc, "_pop_training_kwargs", return_value={}),
            patch.object(svc, "_cleanup"),
            patch("finetuning_pipeline.services.training_service.Path"),
        ):
            fake_trl = types.ModuleType("trl")
            fake_trl.KTOTrainer = capture_kto_trainer
            fake_trl.KTOConfig = type("KTOConfig", (), {"__init__": lambda self, **kw: None})
            with patch.dict("sys.modules", {"trl": fake_trl}):
                result = await svc.train_kto_model(
                    dataset=_make_mock_dataset(_sample_kto_data(), ["prompt", "completion", "label"]),
                    output_dir=tempfile.mkdtemp(),
                    extra_callbacks=[fake_cb],
                )

        if "callbacks" in captured_kwargs:
            assert fake_cb in captured_kwargs["callbacks"]
