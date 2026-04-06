"""Unit tests for the finetuning training service enhancements.

Tests verify SFTConfig usage, completion_only_loss, prepare_model_for_kbit_training,
early stopping callback, report_to, eval file loading, and push_to_hub.

NOTE: We avoid patching sys.modules for installed packages (datasets, transformers,
numpy, trl) because numpy's C extension cannot be reloaded in-process. Instead we
use builtins.__import__ interception and patch.object on service methods.
"""
from __future__ import annotations

import builtins
import inspect
import json
import os
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finetuning_pipeline.services.training_service import TrainingService


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_mock_dataset(rows: List[Dict[str, str]]):
    """Create a mock HF Dataset with column_names and map/remove_columns."""
    dataset = MagicMock()
    dataset.column_names = list(rows[0].keys()) if rows else []
    dataset.__len__ = lambda self: len(rows)
    dataset.__iter__ = lambda self: iter(rows)
    dataset.__getitem__ = lambda self, idx: rows[idx]

    def mock_map(fn, **kwargs):
        new_rows = [{**row, **fn(row)} for row in rows]
        return _make_mock_dataset(new_rows)

    def mock_remove_columns(cols):
        new_rows = [{k: v for k, v in row.items() if k not in cols} for row in rows]
        return _make_mock_dataset(new_rows)

    dataset.map = mock_map
    dataset.remove_columns = mock_remove_columns
    return dataset


def _sample_sft_data():
    return [
        {"prompt": "What is Python?", "response": "Python is a programming language."},
        {"prompt": "What is 2+2?", "response": "4"},
    ]


def _sample_chat_triplet_data():
    return [
        {
            "system": "You are a concise WhatsApp sales assistant.",
            "user": "Salestify itu apa?",
            "assistant": "Salestify membantu bisnis menangani chat WhatsApp lebih cepat.",
        },
        {
            "system": "Tetap singkat dan sopan.",
            "user": "Bisa bantu follow-up customer?",
            "assistant": "Bisa, supaya follow-up lebih konsisten dan cepat.",
        },
    ]


def _sample_grpo_data():
    return [
        {
            "prompt": "Salestify itu apa?",
            "responses": [
                "Salestify membantu bisnis menangani chat WhatsApp.",
                "Salestify adalah aplikasi game.",
            ],
            "rewards": [1.0, 0.0],
        }
    ]


def _sample_dpo_data():
    return [
        {
            "prompt": "Salestify itu apa?",
            "chosen": "Salestify membantu bisnis menangani chat WhatsApp lebih rapi.",
            "rejected": "Salestify adalah aplikasi broadcast biasa.",
        }
    ]


def _sample_kto_data():
    return [
        {
            "prompt": "Salestify itu apa?",
            "completion": "Salestify membantu tim sales menangani chat WhatsApp lebih rapi.",
            "label": True,
        },
        {
            "prompt": "Salestify itu apa?",
            "completion": "Salestify cuma aplikasi broadcast biasa.",
            "label": False,
        },
    ]


def _make_fake_sft_config(**extra_params):
    """Create a fake SFTConfig class with configurable __init__ signature."""
    def init(self, **kwargs):
        pass
    cls = type("SFTConfig", (), {"__init__": init})
    params = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
        inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
    ]
    for name, default in extra_params.items():
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default))
    cls.__init__.__signature__ = inspect.Signature(parameters=params)
    return cls


def _make_fake_sft_trainer(captured_kwargs: dict = None):
    """Create a fake SFTTrainer class that captures __init__ kwargs."""
    def init(self, **kwargs):
        if captured_kwargs is not None:
            captured_kwargs.update(kwargs)
    cls = type("SFTTrainer", (), {
        "__init__": init,
        "train": MagicMock(),
        "save_model": MagicMock(),
    })
    params = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("eval_dataset", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("dataset_text_field", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("max_seq_length", inspect.Parameter.KEYWORD_ONLY, default=2048),
        inspect.Parameter("packing", inspect.Parameter.KEYWORD_ONLY, default=False),
        inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
    cls.__init__.__signature__ = inspect.Signature(parameters=params)
    return cls


def _mock_model_and_tokenizer():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.pad_token = "pad"
    mock_tokenizer.apply_chat_template = MagicMock(return_value="text")
    mock_tokenizer.save_pretrained = MagicMock()
    mock_model.push_to_hub = MagicMock()
    mock_tokenizer.push_to_hub = MagicMock()
    return mock_model, mock_tokenizer


@contextmanager
def _intercept_trl_import(fake_sft_config, fake_sft_trainer):
    """Intercept 'from trl import SFTConfig, SFTTrainer' without touching sys.modules.

    This avoids the numpy C-extension reload crash that happens when
    sys.modules["trl"] or sys.modules["datasets"] are replaced.
    """
    real_import = builtins.__import__

    # Build a fake trl module
    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = fake_sft_config
    fake_trl.SFTTrainer = fake_sft_trainer

    def patched_import(name, *args, **kwargs):
        if name == "trl":
            return fake_trl
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=patched_import):
        yield


@contextmanager
def _intercept_peft_import(fake_peft_module):
    """Intercept runtime PEFT imports used by training helpers."""
    real_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "peft":
            return fake_peft_module
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=patched_import):
        yield


@contextmanager
def _intercept_imports(module_map):
    """Intercept selected runtime imports without mutating sys.modules."""
    real_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name in module_map:
            return module_map[name]
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=patched_import):
        yield


# ──────────────────────────────────────────────
# _pop_training_kwargs tests
# ──────────────────────────────────────────────

class TestPopTrainingKwargs:
    def test_default_report_to_is_empty_list(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({}, cuda_available=False, bf16_supported=False)
        assert result["report_to"] == []

    def test_report_to_string_converted_to_list(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({"report_to": "wandb"}, cuda_available=False, bf16_supported=False)
        assert result["report_to"] == ["wandb"]

    def test_report_to_list_passthrough(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({"report_to": ["wandb", "tensorboard"]}, cuda_available=False, bf16_supported=False)
        assert result["report_to"] == ["wandb", "tensorboard"]

    def test_lr_scheduler_type_default_linear(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({}, cuda_available=False, bf16_supported=False)
        assert result["lr_scheduler_type"] == "linear"

    def test_lr_scheduler_type_cosine(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({"lr_scheduler_type": "cosine"}, cuda_available=False, bf16_supported=False)
        assert result["lr_scheduler_type"] == "cosine"

    def test_warmup_ratio_default_zero(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({}, cuda_available=False, bf16_supported=False)
        assert result["warmup_ratio"] == 0.0

    def test_warmup_ratio_custom(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({"warmup_ratio": 0.1}, cuda_available=False, bf16_supported=False)
        assert result["warmup_ratio"] == 0.1

    def test_warmup_steps_and_max_steps_custom(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs(
            {"warmup_steps": 5, "max_steps": 60},
            cuda_available=False,
            bf16_supported=False,
        )
        assert result["warmup_steps"] == 5
        assert result["max_steps"] == 60

    def test_weight_decay_and_max_grad_norm(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({"weight_decay": 0.01, "max_grad_norm": 0.3}, cuda_available=False, bf16_supported=False)
        assert result["weight_decay"] == 0.01
        assert result["max_grad_norm"] == 0.3

    def test_seed_default(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({}, cuda_available=False, bf16_supported=False)
        assert result["seed"] == 42

    def test_per_device_eval_batch_size(self):
        svc = TrainingService()
        result = svc._pop_training_kwargs({"per_device_eval_batch_size": 4}, cuda_available=False, bf16_supported=False)
        assert result["per_device_eval_batch_size"] == 4


# ──────────────────────────────────────────────
# train_model — SFTConfig usage
# ──────────────────────────────────────────────

class TestTrainModelSFTConfig:
    def test_apply_lora_to_model_wraps_and_casts_trainable_params(self):
        svc = TrainingService()
        import torch

        param = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
        frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)
        wrapped_model = MagicMock()
        wrapped_model.named_parameters.return_value = [
            ("adapter", param),
            ("frozen", frozen),
        ]
        fake_peft = types.ModuleType("peft")
        fake_peft.get_peft_model = MagicMock(return_value=wrapped_model)

        with _intercept_peft_import(fake_peft):
            result_model, trainer_peft_config, cast_count = svc._apply_lora_to_model(
                model=MagicMock(),
                peft_config=MagicMock(),
            )

        assert result_model is wrapped_model
        assert trainer_peft_config is None
        assert cast_count == 1
        assert param.dtype == torch.float32
        assert frozen.dtype == torch.float16

    def test_apply_lora_to_model_falls_back_when_peft_wrap_unavailable(self):
        svc = TrainingService()
        model = MagicMock()
        peft_config = MagicMock()

        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "peft":
                raise ImportError("missing peft")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=patched_import):
            result_model, trainer_peft_config, cast_count = svc._apply_lora_to_model(
                model=model,
                peft_config=peft_config,
            )

        assert result_model is model
        assert trainer_peft_config is peft_config
        assert cast_count == 0

    @pytest.mark.asyncio
    async def test_sft_config_receives_completion_only_loss(self):
        """Verify completion_only_loss is passed to _build_config as extra_kwargs."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        captured_build_config_calls: list = []

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        fake_config = _make_fake_sft_config(completion_only_loss=False)
        fake_trainer = _make_fake_sft_trainer()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_sft",
                completion_only_loss=True,
            )

        assert len(captured_build_config_calls) == 1
        extra = captured_build_config_calls[0]["kwargs"].get("extra_kwargs", {})
        assert extra.get("completion_only_loss") is True

    @pytest.mark.asyncio
    async def test_sft_config_receives_notebook_parity_training_knobs(self):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_build_config_calls: list = []

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        fake_config = _make_fake_sft_config(
            completion_only_loss=False,
            dataset_text_field=None,
            max_length=2048,
            packing=False,
        )
        fake_trainer = _make_fake_sft_trainer()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_sft_parity_knobs",
                max_steps=60,
                warmup_steps=5,
                logging_steps=1,
                save_steps=30,
                seed=3407,
                max_seq_length=384,
                packing=False,
            )

        build_call = captured_build_config_calls[0]["kwargs"]
        training_kwargs = build_call["training_kwargs"]
        extra = build_call.get("extra_kwargs", {})
        assert training_kwargs["max_steps"] == 60
        assert training_kwargs["warmup_steps"] == 5
        assert training_kwargs["logging_steps"] == 1
        assert training_kwargs["save_steps"] == 30
        assert training_kwargs["seed"] == 3407
        assert extra["dataset_text_field"] == "text"
        assert extra["max_length"] == 384
        assert extra["packing"] is False

    @pytest.mark.asyncio
    async def test_train_model_seeds_before_loading_model(self, tmp_path):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_order: list = []

        def record_seed(seed):
            captured_order.append(("seed", seed))

        def record_load(*args, **kwargs):
            captured_order.append(("load", args[0]))
            return mock_model, mock_tokenizer

        fake_config = _make_fake_sft_config(completion_only_loss=False)
        fake_trainer = _make_fake_sft_trainer()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_set_global_seed", side_effect=record_seed),
            patch.object(svc, "_load_model_and_tokenizer", side_effect=record_load),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_seed_before_load"),
                seed=3407,
            )

        assert captured_order[:2] == [
            ("seed", 3407),
            ("load", svc.config.base_model),
        ]

    @pytest.mark.asyncio
    async def test_sft_config_disables_mixed_precision_for_notebook_chat_triplet_path(self):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_chat_triplet_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_build_config_calls: list = []

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        fake_config = _make_fake_sft_config(
            completion_only_loss=False,
            dataset_text_field=None,
            max_length=2048,
            packing=False,
        )
        fake_trainer = _make_fake_sft_trainer()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(True, True)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_sft_chat_triplet_precision",
                completion_only_loss=False,
                load_in_4bit=True,
            )

        training_kwargs = captured_build_config_calls[0]["kwargs"]["training_kwargs"]
        assert training_kwargs["fp16"] is False
        assert training_kwargs["bf16"] is False

    def test_prepare_sft_dataset_includes_system_prompt_when_provided(self):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        _mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        captured_messages: list = []

        def capture_template(messages, tokenize=False):
            captured_messages.append(messages)
            return "templated"

        mock_tokenizer.apply_chat_template = MagicMock(side_effect=capture_template)

        prepared = svc._prepare_sft_text_dataset(
            dataset=dataset,
            tokenizer=mock_tokenizer,
            prompt_column="prompt",
            response_column="response",
            system_prompt="Use <think> tags when reasoning is useful.",
        )

        assert prepared.column_names == ["prompt", "completion", "text"]
        assert captured_messages
        assert captured_messages[0][0]["role"] == "system"
        assert "<think>" in captured_messages[0][0]["content"]

    def test_prepare_sft_dataset_supports_chat_triplet_rows(self):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_chat_triplet_data())
        _mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        captured_messages: list = []

        def capture_template(messages, tokenize=False):
            captured_messages.append(messages)
            return "templated"

        mock_tokenizer.apply_chat_template = MagicMock(side_effect=capture_template)

        prepared = svc._prepare_sft_text_dataset(
            dataset=dataset,
            tokenizer=mock_tokenizer,
            prompt_column="prompt",
            response_column="response",
            system_prompt="Jawab singkat.",
            system_column="system",
            user_column="user",
            assistant_column="assistant",
        )

        assert prepared.column_names == ["prompt", "completion", "text"]
        assert captured_messages
        assert [message["role"] for message in captured_messages[0]] == [
            "system",
            "user",
            "assistant",
        ]
        assert "Jawab singkat." in captured_messages[0][0]["content"]
        assert "You are a concise WhatsApp sales assistant." in captured_messages[0][0]["content"]

    def test_prepare_sft_dataset_supports_text_only_notebook_mode(self):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_chat_triplet_data())
        _mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        prepared = svc._prepare_sft_text_dataset(
            dataset=dataset,
            tokenizer=mock_tokenizer,
            prompt_column="prompt",
            response_column="response",
            system_prompt=None,
            system_column="system",
            user_column="user",
            assistant_column="assistant",
            text_only=True,
        )

        assert prepared.column_names == ["text"]

    @pytest.mark.asyncio
    async def test_train_model_accepts_chat_triplet_dataset(self, tmp_path):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_chat_triplet_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_trainer_kwargs: dict = {}

        fake_config = _make_fake_sft_config(completion_only_loss=False)
        fake_trainer = _make_fake_sft_trainer(captured_trainer_kwargs)

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_sft_chat_triplet"),
                system_prompt="Prioritaskan pain point bisnis.",
            )

        train_dataset = captured_trainer_kwargs["train_dataset"]
        assert set(train_dataset.column_names) == {"prompt", "completion", "text"}

    @pytest.mark.asyncio
    async def test_train_model_uses_text_only_dataset_for_chat_triplet_notebook_mode(self, tmp_path):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_chat_triplet_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_trainer_kwargs: dict = {}

        fake_config = _make_fake_sft_config(completion_only_loss=False)
        fake_trainer = _make_fake_sft_trainer(captured_trainer_kwargs)

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
        ):
            result = await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_sft_chat_triplet_text_only"),
                completion_only_loss=False,
            )

        train_dataset = captured_trainer_kwargs["train_dataset"]
        assert set(train_dataset.column_names) == {"text"}
        assert result["config"]["dataset_format"] == "text_only"

    @pytest.mark.asyncio
    async def test_train_model_prefers_explicit_lora_wrapping(self, tmp_path):
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        wrapped_model = MagicMock()
        captured_trainer_kwargs: dict = {}

        fake_config = _make_fake_sft_config(completion_only_loss=False)
        fake_trainer = _make_fake_sft_trainer(captured_trainer_kwargs)

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_apply_lora_to_model", return_value=(wrapped_model, None, 3)),
        ):
            result = await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_sft_explicit_lora"),
            )

        assert captured_trainer_kwargs["model"] is wrapped_model
        assert "peft_config" not in captured_trainer_kwargs
        assert result["config"]["lora_trainable_fp32_tensors"] == 3

    def test_register_special_tokens_resizes_embeddings(self):
        svc = TrainingService()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.add_special_tokens.return_value = 2
        mock_tokenizer.__len__.return_value = 321
        mock_model.config = MagicMock()

        applied = svc._register_special_tokens(
            model=mock_model,
            tokenizer=mock_tokenizer,
            special_tokens=["<think>", "</think>"],
        )

        assert applied == ["<think>", "</think>"]
        mock_tokenizer.add_special_tokens.assert_called_once_with(
            {"additional_special_tokens": ["<think>", "</think>"]}
        )
        mock_model.resize_token_embeddings.assert_called_once_with(321)

    @pytest.mark.asyncio
    async def test_sft_dataset_keeps_prompt_completion_and_text_columns(self, tmp_path):
        """Normalize SFT rows to a schema that works across TRL variants."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_trainer_kwargs: dict = {}

        fake_config = _make_fake_sft_config(completion_only_loss=False)
        fake_trainer = _make_fake_sft_trainer(captured_trainer_kwargs)

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_sft_columns"),
            )

        train_dataset = captured_trainer_kwargs["train_dataset"]
        assert set(train_dataset.column_names) == {"prompt", "completion", "text"}

    @pytest.mark.asyncio
    async def test_sft_result_includes_compact_metric_summary(self, tmp_path):
        """Expose final train/eval metrics in the training result payload."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        def init(self, **kwargs):
            self.state = types.SimpleNamespace(
                log_history=[
                    {"loss": 1.2, "learning_rate": 2e-4, "step": 1},
                    {"eval_loss": 0.9, "step": 1},
                    {"loss": 0.8, "grad_norm": 0.4, "step": 2},
                ],
                best_metric=0.9,
                global_step=2,
                epoch=1.0,
            )

        def train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.75)

        fake_trainer = type("SFTTrainer", (), {
            "__init__": init,
            "train": train,
            "save_model": MagicMock(),
        })
        fake_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("eval_dataset", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("dataset_text_field", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("max_seq_length", inspect.Parameter.KEYWORD_ONLY, default=2048),
            inspect.Parameter("packing", inspect.Parameter.KEYWORD_ONLY, default=False),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])
        fake_config = _make_fake_sft_config(completion_only_loss=False)

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
        ):
            result = await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_sft_metrics"),
            )

        assert result["metrics"]["training_loss"] == 0.75
        assert result["metrics"]["eval_loss"] == 0.9
        assert result["metrics"]["best_eval_loss"] == 0.9
        assert result["metrics"]["global_step"] == 2

    @pytest.mark.asyncio
    async def test_retries_without_completion_only_loss_when_trl_expects_completion_column(self, tmp_path):
        """Fallback for TRL stacks that raise KeyError('completion') during SFT setup."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        captured_build_config_calls: list = []
        trainer_init_calls = {"count": 0}

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        def flaky_init(self, **kwargs):
            trainer_init_calls["count"] += 1
            if trainer_init_calls["count"] == 1:
                raise KeyError("completion")

        fake_trainer = type("SFTTrainer", (), {
            "__init__": flaky_init,
            "train": MagicMock(),
            "save_model": MagicMock(),
        })
        fake_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("eval_dataset", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("dataset_text_field", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("max_seq_length", inspect.Parameter.KEYWORD_ONLY, default=2048),
            inspect.Parameter("packing", inspect.Parameter.KEYWORD_ONLY, default=False),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])
        fake_config = _make_fake_sft_config(completion_only_loss=False)

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            result = await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_sft_retry"),
                completion_only_loss=True,
            )

        assert result["success"] is True
        assert trainer_init_calls["count"] == 2
        assert "warnings" in result
        assert "completion_only_loss" in result["warnings"][0]
        assert result["config"]["completion_only_loss_requested"] is True
        assert result["config"]["completion_only_loss_effective"] is False
        assert len(captured_build_config_calls) == 2
        assert captured_build_config_calls[0]["kwargs"].get("extra_kwargs", {}).get("completion_only_loss") is True
        assert captured_build_config_calls[1]["kwargs"].get("extra_kwargs", {}).get("completion_only_loss") is False

    @pytest.mark.asyncio
    async def test_retries_without_completion_only_loss_when_trl_rejects_formatting_func(self, tmp_path):
        """Fallback for TRL stacks that reject formatted datasets with completion-only loss."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        captured_build_config_calls: list = []
        trainer_init_calls = {"count": 0}

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        def flaky_init(self, **kwargs):
            trainer_init_calls["count"] += 1
            if trainer_init_calls["count"] == 1:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, "
                    "which is incompatible. Using a formatter converts the dataset to a "
                    "language modeling type, conflicting with completion-only loss."
                )

        fake_trainer = type("SFTTrainer", (), {
            "__init__": flaky_init,
            "train": MagicMock(),
            "save_model": MagicMock(),
        })
        fake_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("eval_dataset", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("formatting_func", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("max_seq_length", inspect.Parameter.KEYWORD_ONLY, default=2048),
            inspect.Parameter("packing", inspect.Parameter.KEYWORD_ONLY, default=False),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])
        fake_config = _make_fake_sft_config(completion_only_loss=False)

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            result = await svc.train_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_sft_retry_formatting_func"),
                completion_only_loss=True,
            )

        assert result["success"] is True
        assert trainer_init_calls["count"] == 2
        assert "warnings" in result
        assert "completion_only_loss" in result["warnings"][0]
        assert result["config"]["completion_only_loss_requested"] is True
        assert result["config"]["completion_only_loss_effective"] is False
        assert len(captured_build_config_calls) == 2
        assert captured_build_config_calls[0]["kwargs"].get("extra_kwargs", {}).get("completion_only_loss") is True
        assert captured_build_config_calls[1]["kwargs"].get("extra_kwargs", {}).get("completion_only_loss") is False

    def test_train_model_source_uses_sft_config(self):
        """Verify source code imports SFTConfig (not TrainingArguments) for SFT."""
        source = inspect.getsource(TrainingService.train_model)
        assert "from trl import SFTConfig" in source
        assert "TrainingArguments" not in source


# ──────────────────────────────────────────────
# prepare_model_for_kbit_training
# ──────────────────────────────────────────────

class TestPrepareModelForKbitTraining:
    def test_code_calls_prepare_when_quantization_config_present(self):
        """Verify the source contains prepare_model_for_kbit_training guarded by quantization check."""
        source = inspect.getsource(TrainingService._load_model_and_tokenizer)
        assert "prepare_model_for_kbit_training" in source
        assert "if quantization_config is not None" in source

    def test_no_prepare_when_load_in_4bit_false(self):
        """When load_in_4bit=False, quantization_config stays None."""
        kwargs = {"load_in_4bit": False}
        load_in_4bit = bool(kwargs.pop("load_in_4bit", True))
        assert load_in_4bit is False

    def test_build_quantization_config_uses_fp16_like_notebook(self):
        """Notebook parity keeps 4-bit compute dtype on fp16 rather than bf16."""
        svc = TrainingService()
        fake_transformers = types.ModuleType("transformers")
        captured_kwargs = {}

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
        fake_bitsandbytes = types.ModuleType("bitsandbytes")
        fake_torch = types.SimpleNamespace(float16="fp16", float32="fp32")

        with (
            _intercept_imports({
                "transformers": fake_transformers,
                "bitsandbytes": fake_bitsandbytes,
            }),
            patch.object(svc, "_preflight_bnb_check", return_value=True),
        ):
            config = svc._build_quantization_config(True, fake_torch)

        assert config is not None
        assert captured_kwargs["bnb_4bit_compute_dtype"] == "fp16"


class TestModelSaving:
    def test_save_model_artifacts_prefers_trainer_model_save_pretrained(self):
        trainer = MagicMock()
        trainer.model = MagicMock()
        trainer.model.save_pretrained = MagicMock()
        trainer.save_model = MagicMock()
        tokenizer = MagicMock()

        TrainingService._save_model_artifacts(trainer, tokenizer, "/tmp/out")

        trainer.model.save_pretrained.assert_called_once_with("/tmp/out")
        trainer.save_model.assert_not_called()
        tokenizer.save_pretrained.assert_called_once_with("/tmp/out")


# ──────────────────────────────────────────────
# Early stopping callback
# ──────────────────────────────────────────────

class TestEarlyStopping:
    @pytest.mark.asyncio
    async def test_early_stopping_callback_added_when_patience_set(self):
        """Verify EarlyStoppingCallback is created when early_stopping_patience is set."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        eval_dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        captured_trainer_kwargs: dict = {}
        fake_trainer = _make_fake_sft_trainer(captured_trainer_kwargs)
        fake_config = _make_fake_sft_config(completion_only_loss=False)

        mock_es_instance = MagicMock()
        mock_es_cls = MagicMock(return_value=mock_es_instance)

        # Intercept the EarlyStoppingCallback import
        real_import = builtins.__import__
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTConfig = fake_config
        fake_trl.SFTTrainer = fake_trainer

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.EarlyStoppingCallback = mock_es_cls

        def patched_import(name, *args, **kwargs):
            if name == "trl":
                return fake_trl
            if name == "transformers":
                return fake_transformers
            return real_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=patched_import),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_es",
                evaluation_dataset=eval_dataset,
                early_stopping_patience=3,
            )

        mock_es_cls.assert_called_once_with(early_stopping_patience=3)
        assert "callbacks" in captured_trainer_kwargs
        assert mock_es_instance in captured_trainer_kwargs["callbacks"]

    @pytest.mark.asyncio
    async def test_no_callback_when_patience_none(self):
        """No EarlyStoppingCallback when early_stopping_patience is None."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        captured_trainer_kwargs: dict = {}
        fake_trainer = _make_fake_sft_trainer(captured_trainer_kwargs)
        fake_config = _make_fake_sft_config()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_no_es",
                early_stopping_patience=None,
            )

        assert "callbacks" not in captured_trainer_kwargs


# ──────────────────────────────────────────────
# Eval file loading
# ──────────────────────────────────────────────

class TestEvalFileLoading:
    @pytest.mark.asyncio
    async def test_eval_file_path_triggers_load(self):
        """Verify that eval_file_path causes load_dataset_from_file to be called."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_eval_dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        fake_config = _make_fake_sft_config()
        fake_trainer = _make_fake_sft_trainer()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
            patch.object(
                svc, "load_dataset_from_file",
                new_callable=AsyncMock,
                return_value={"success": True, "dataset_object": mock_eval_dataset},
            ) as mock_load,
        ):
            await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_eval",
                eval_file_path="/data/eval.jsonl",
            )

        mock_load.assert_called_once_with("/data/eval.jsonl", format="jsonl")


# ──────────────────────────────────────────────
# load_dataset_from_file
# ──────────────────────────────────────────────

class TestLoadDatasetFromFile:
    @pytest.mark.asyncio
    async def test_load_jsonl_merges_instruction_input(self):
        """Verify instruction+input merged into prompt, output renamed to response."""
        svc = TrainingService()

        sample_data = [
            {"instruction": "Explain X", "input": "context here", "output": "X is..."},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
            tmp_path = f.name

        try:
            result = await svc.load_dataset_from_file(tmp_path, format="jsonl")
            assert result["success"] is True
            assert "prompt" in result["columns"]
            assert "response" in result["columns"]
            assert "instruction" not in result["columns"]
            assert "input" not in result["columns"]
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_load_json_format(self):
        svc = TrainingService()

        sample_data = [
            {"instruction": "Do Y", "input": "", "output": "Done Y"},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_data, f)
            tmp_path = f.name

        try:
            result = await svc.load_dataset_from_file(tmp_path, format="json")
            assert result["success"] is True
            assert result["num_examples"] == 1
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        svc = TrainingService()
        result = await svc.load_dataset_from_file("/nonexistent/path.jsonl", format="jsonl")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ──────────────────────────────────────────────
# _build_config
# ──────────────────────────────────────────────

class TestBuildConfig:
    def test_eval_strategy_set_when_has_eval(self):
        svc = TrainingService()
        FakeConfig = _make_fake_sft_config()
        captured: dict = {}
        original_sig = FakeConfig.__init__.__signature__

        def spy_init(self, **kwargs):
            captured.update(kwargs)

        FakeConfig.__init__ = spy_init
        FakeConfig.__init__.__signature__ = original_sig

        svc._build_config(
            FakeConfig,
            output_dir="/tmp",
            num_epochs=1,
            has_eval=True,
            save_best_model=True,
            training_kwargs={"report_to": []},
        )

        assert captured["eval_strategy"] == "steps"
        assert captured["load_best_model_at_end"] is True
        assert captured["metric_for_best_model"] == "eval_loss"

    def test_extra_kwargs_merged(self):
        svc = TrainingService()
        FakeConfig = _make_fake_sft_config(completion_only_loss=False)
        captured: dict = {}
        original_sig = FakeConfig.__init__.__signature__

        def spy_init(self, **kwargs):
            captured.update(kwargs)

        FakeConfig.__init__ = spy_init
        FakeConfig.__init__.__signature__ = original_sig

        svc._build_config(
            FakeConfig,
            output_dir="/tmp",
            num_epochs=1,
            has_eval=False,
            save_best_model=False,
            training_kwargs={"report_to": []},
            extra_kwargs={"completion_only_loss": True},
        )

        assert captured["completion_only_loss"] is True


# ──────────────────────────────────────────────
# Push to hub
# ──────────────────────────────────────────────

class TestResolveModelPath:
    def test_resolves_hf_cache_wrapper_to_latest_snapshot(self, tmp_path: Path):
        svc = TrainingService()
        wrapper = tmp_path / "models--demo--model"
        snapshots = wrapper / "snapshots"
        snapshots.mkdir(parents=True)

        older = snapshots / "old-snap"
        older.mkdir()
        newer = snapshots / "new-snap"
        newer.mkdir()

        os.utime(older, (1, 1))
        os.utime(newer, None)

        assert svc._resolve_model_path(str(wrapper)) == str(newer)

    def test_keeps_real_model_directory_unchanged(self, tmp_path: Path):
        svc = TrainingService()
        model_dir = tmp_path / "real-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        assert svc._resolve_model_path(str(model_dir)) == str(model_dir)


class TestPushToHub:
    @pytest.mark.asyncio
    async def test_push_to_hub_calls_model_and_tokenizer(self):
        """Verify push_to_hub is called on model and tokenizer after training."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        fake_config = _make_fake_sft_config()
        fake_trainer = _make_fake_sft_trainer()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
        ):
            result = await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_hub",
                push_to_hub="my-org/my-model",
            )

        mock_model.push_to_hub.assert_called_once_with("my-org/my-model")
        mock_tokenizer.push_to_hub.assert_called_once_with("my-org/my-model")
        assert result.get("hub_url") == "https://huggingface.co/my-org/my-model"

    @pytest.mark.asyncio
    async def test_no_push_when_not_requested(self):
        """Verify push_to_hub is NOT called when push_to_hub is None."""
        svc = TrainingService()
        dataset = _make_mock_dataset(_sample_sft_data())
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()

        fake_config = _make_fake_sft_config()
        fake_trainer = _make_fake_sft_trainer()

        with (
            _intercept_trl_import(fake_config, fake_trainer),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", return_value=MagicMock()),
        ):
            result = await svc.train_model(
                dataset=dataset,
                output_dir="/tmp/test_no_hub",
            )

        mock_model.push_to_hub.assert_not_called()
        mock_tokenizer.push_to_hub.assert_not_called()
        assert "hub_url" not in result


class TestTrainGrpoModel:
    @pytest.mark.asyncio
    async def test_train_dpo_model_can_continue_from_existing_lora_adapter(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_dpo_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        resumed_model = MagicMock()
        resumed_tokenizer = MagicMock()
        captured_trainer_kwargs: dict = {}

        def dpo_config_init(self, **kwargs):
            pass

        fake_dpo_config = type("DPOConfig", (), {"__init__": dpo_config_init})
        fake_dpo_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("beta", inspect.Parameter.KEYWORD_ONLY, default=0.1),
            inspect.Parameter("max_prompt_length", inspect.Parameter.KEYWORD_ONLY, default=512),
            inspect.Parameter("max_length", inspect.Parameter.KEYWORD_ONLY, default=1024),
        ])

        def trainer_init(self, **kwargs):
            captured_trainer_kwargs.update(kwargs)
            self.model = kwargs["model"]

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.12)

        fake_dpo_trainer = type("DPOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_dpo_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.DPOConfig = fake_dpo_config
        fake_trl.DPOTrainer = fake_dpo_trainer

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(
                svc,
                "_load_existing_lora_adapter",
                return_value=(resumed_model, resumed_tokenizer, 7),
            ) as load_existing,
        ):
            result = await svc.train_dpo_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_dpo_continue_adapter"),
                use_lora=True,
                adapter_path="/models/best_sft",
            )

        assert result["success"] is True
        load_existing.assert_called_once()
        assert captured_trainer_kwargs["model"] is resumed_model
        assert "peft_config" not in captured_trainer_kwargs
        assert result["config"]["continued_from_adapter"] is True
        assert result["config"]["adapter_path"] == "/models/best_sft"
        assert result["config"]["lora_trainable_fp32_tensors"] == 7

    @pytest.mark.asyncio
    async def test_train_dpo_model_uses_notebook_parity_lengths_and_training_defaults(self, tmp_path):
        svc = TrainingService()
        dataset = [
            {
                "prompt": "Salestify itu apa?\n\n",
                "chosen": " Salestify membantu bisnis menangani chat WhatsApp lebih rapi. ",
                "rejected": "Salestify adalah aplikasi broadcast biasa.\t",
                "metadata": {"source": "unit-test"},
            }
        ]
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        mock_tokenizer.side_effect = (
            lambda text, add_special_tokens=False, return_attention_mask=False: {
                "input_ids": list(range(1, len(str(text).split()) + 1))
            }
        )
        captured_build_config_calls: list = []
        captured_trainer_kwargs: dict = {}

        def dpo_config_init(self, **kwargs):
            pass

        fake_dpo_config = type("DPOConfig", (), {"__init__": dpo_config_init})
        fake_dpo_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("beta", inspect.Parameter.KEYWORD_ONLY, default=0.1),
            inspect.Parameter("max_prompt_length", inspect.Parameter.KEYWORD_ONLY, default=512),
            inspect.Parameter("max_length", inspect.Parameter.KEYWORD_ONLY, default=1024),
            inspect.Parameter("max_completion_length", inspect.Parameter.KEYWORD_ONLY, default=256),
        ])

        def trainer_init(self, **kwargs):
            captured_trainer_kwargs.update(kwargs)
            self.model = kwargs["model"]
            self.state = types.SimpleNamespace(log_history=[], global_step=0, epoch=0)

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.09)

        fake_dpo_trainer = type("DPOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_dpo_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.DPOConfig = fake_dpo_config
        fake_trl.DPOTrainer = fake_dpo_trainer

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            return types.SimpleNamespace()

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(True, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            result = await svc.train_dpo_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_dpo_parity_defaults"),
                gradient_checkpointing=True,
            )

        assert result["success"] is True
        assert captured_build_config_calls[0]["kwargs"]["num_epochs"] == 1
        training_kwargs = captured_build_config_calls[0]["kwargs"]["training_kwargs"]
        extra = captured_build_config_calls[0]["kwargs"]["extra_kwargs"]
        assert training_kwargs["learning_rate"] == 1e-4
        assert training_kwargs["weight_decay"] == 0.01
        assert training_kwargs["max_grad_norm"] == 0.0
        assert training_kwargs["bf16"] is False
        assert training_kwargs["fp16"] is False
        assert training_kwargs["gradient_checkpointing"] is True
        assert extra["beta"] == 0.1
        assert extra["max_prompt_length"] == 384
        assert extra["max_length"] == 512
        assert extra["max_completion_length"] == 128
        assert captured_trainer_kwargs["processing_class"] is mock_tokenizer
        assert "dpo_preprocessing" in result["artifacts"]
        assert "preference_normalization" in result["artifacts"]
        assert "dpo_trainer_dataset" in result["artifacts"]
        normalization = json.loads(
            Path(result["artifacts"]["preference_normalization"]).read_text(encoding="utf-8")
        )
        preprocessing = json.loads(
            Path(result["artifacts"]["dpo_preprocessing"]).read_text(encoding="utf-8")
        )
        trainer_dataset = json.loads(
            Path(result["artifacts"]["dpo_trainer_dataset"]).read_text(encoding="utf-8")
        )
        assert normalization["trimmed_row_count"] == 1
        assert normalization["trimmed_scalar_value_count"] == 3
        assert preprocessing["max_prompt_length"] == 384
        assert preprocessing["max_length"] == 512
        assert preprocessing["sample_size"] == 1
        assert trainer_dataset["num_rows"] == len(dataset)
        assert "metadata" in trainer_dataset["column_names"]
        assert "prompt" in trainer_dataset["column_names"]
        assert result["config"]["auto_tuned_defaults"]["applied"] is True
        assert result["config"]["auto_tuned_defaults"]["effective"]["num_epochs"] == 1

    @pytest.mark.asyncio
    async def test_train_dpo_model_preserves_custom_small_dataset_budget(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_dpo_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_build_config_calls: list = []

        def dpo_config_init(self, **kwargs):
            pass

        fake_dpo_config = type("DPOConfig", (), {"__init__": dpo_config_init})
        fake_dpo_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("beta", inspect.Parameter.KEYWORD_ONLY, default=0.1),
        ])

        def trainer_init(self, **kwargs):
            self.model = kwargs["model"]

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.09)

        fake_dpo_trainer = type("DPOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_dpo_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.DPOConfig = fake_dpo_config
        fake_trl.DPOTrainer = fake_dpo_trainer

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            return types.SimpleNamespace()

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            result = await svc.train_dpo_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_dpo_custom_budget"),
                num_epochs=2,
                learning_rate=5e-5,
            )

        assert result["success"] is True
        assert captured_build_config_calls[0]["kwargs"]["num_epochs"] == 2
        training_kwargs = captured_build_config_calls[0]["kwargs"]["training_kwargs"]
        assert training_kwargs["learning_rate"] == 5e-5
        assert result["config"]["auto_tuned_defaults"]["applied"] is False

    @pytest.mark.asyncio
    async def test_train_dpo_model_seeds_before_loading_model(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_dpo_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_order: list = []

        def dpo_config_init(self, **kwargs):
            pass

        fake_dpo_config = type("DPOConfig", (), {"__init__": dpo_config_init})
        fake_dpo_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("beta", inspect.Parameter.KEYWORD_ONLY, default=0.1),
        ])

        def trainer_init(self, **kwargs):
            self.model = kwargs["model"]

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.09)

        fake_dpo_trainer = type("DPOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_dpo_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.DPOConfig = fake_dpo_config
        fake_trl.DPOTrainer = fake_dpo_trainer

        def record_seed(seed):
            captured_order.append(("seed", seed))

        def record_load(*args, **kwargs):
            captured_order.append(("load", args[0]))
            return mock_model, mock_tokenizer

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_set_global_seed", side_effect=record_seed),
            patch.object(svc, "_load_model_and_tokenizer", side_effect=record_load),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
        ):
            result = await svc.train_dpo_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_dpo_seed_before_load"),
                seed=3407,
            )

        assert result["success"] is True
        assert captured_order[:2] == [
            ("seed", 3407),
            ("load", svc.config.base_model),
        ]

    @pytest.mark.asyncio
    async def test_train_grpo_model_supports_lora_wrapping_and_generation_batch_size(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_grpo_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        wrapped_model = MagicMock()
        captured_build_config_calls: list = []
        captured_trainer_kwargs: dict = {}

        def grpo_config_init(self, **kwargs):
            pass

        fake_grpo_config = type("GRPOConfig", (), {"__init__": grpo_config_init})
        fake_grpo_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("num_generations", inspect.Parameter.KEYWORD_ONLY, default=4),
            inspect.Parameter("max_prompt_length", inspect.Parameter.KEYWORD_ONLY, default=512),
            inspect.Parameter("max_completion_length", inspect.Parameter.KEYWORD_ONLY, default=256),
            inspect.Parameter("generation_batch_size", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("steps_per_generation", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        def trainer_init(self, **kwargs):
            captured_trainer_kwargs.update(kwargs)
            self.model = kwargs["model"]
            self.reward_funcs = kwargs["reward_funcs"]
            self.state = types.SimpleNamespace(
                log_history=[
                    {
                        "reward": 0.25,
                        "completions/clipped_ratio": 0.5,
                        "completions/mean_length": 44.0,
                        "completions/mean_terminated_length": 33.0,
                        "completions/max_length": 48.0,
                        "completions/max_terminated_length": 36.0,
                    },
                    {
                        "reward": 0.0,
                        "completions/clipped_ratio": 1.0,
                        "completions/mean_length": 64.0,
                        "completions/mean_terminated_length": 0.0,
                        "completions/max_length": 64.0,
                        "completions/max_terminated_length": 0.0,
                    },
                ],
                global_step=0,
                epoch=0,
            )

        def trainer_train(self, **kwargs):
            self.reward_funcs[0](
                ["Salestify itu apa?", "Salestify itu apa?"],
                [
                    "Assistant: Salestify membantu bisnis menangani chat WhatsApp. <|eot_id|>",
                    "Cuaca hari ini cerah.",
                ],
                completion_ids=[
                    [101, 2],
                    [201, 202, 203],
                ],
            )
            return types.SimpleNamespace(training_loss=0.25)

        fake_grpo_trainer = type("GRPOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_grpo_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("reward_funcs", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.GRPOConfig = fake_grpo_config
        fake_trl.GRPOTrainer = fake_grpo_trainer

        def spy_build_config(*args, **kwargs):
            captured_build_config_calls.append({"args": args, "kwargs": kwargs})
            extra = kwargs["extra_kwargs"]
            return types.SimpleNamespace(
                generation_batch_size=extra.get("generation_batch_size"),
                steps_per_generation=extra.get("steps_per_generation"),
            )

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_apply_lora_to_model", return_value=(wrapped_model, None, 5)),
            patch.object(svc, "_build_config", side_effect=spy_build_config),
        ):
            result = await svc.train_grpo_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_grpo_lora"),
                use_lora=True,
                num_generations=2,
                generation_batch_size=2,
                steps_per_generation=3,
            )

        assert result["success"] is True
        assert captured_trainer_kwargs["model"] is wrapped_model
        assert "peft_config" not in captured_trainer_kwargs
        assert captured_build_config_calls[0]["kwargs"]["num_epochs"] == 1
        extra = captured_build_config_calls[0]["kwargs"]["extra_kwargs"]
        training_kwargs = captured_build_config_calls[0]["kwargs"]["training_kwargs"]
        assert training_kwargs["learning_rate"] == 1e-4
        assert extra["num_generations"] == 2
        assert extra["generation_batch_size"] == 2
        assert extra["steps_per_generation"] == 3
        assert result["config"]["use_lora"] is True
        assert result["config"]["lora_trainable_fp32_tensors"] == 5
        assert result["config"]["generation_batch_size"] == 2
        assert result["config"]["steps_per_generation"] == 3
        assert result["config"]["auto_tuned_defaults"]["applied"] is True
        reward_stats = result["config"]["reward_match_stats"]
        assert reward_stats["queries"] == 2
        assert reward_stats["exact_matches"] == 1
        assert reward_stats["misses"] == 1
        assert reward_stats["positive_rewards"] == 1
        assert reward_stats["negative_rewards"] == 1
        assert reward_stats["zero_rewards"] == 0
        assert reward_stats["truncation_checks"] == 2
        assert reward_stats["truncated_queries"] == 1
        assert reward_stats["penalized_queries"] == 1
        assert reward_stats["avg_reward"] == 0.425
        assert reward_stats["avg_reward_adjustment"] == -0.075
        training_diagnostics = result["config"]["training_diagnostics"]
        assert training_diagnostics["steps_logged"] == 2
        assert training_diagnostics["positive_reward_steps"] == 1
        assert training_diagnostics["clip_ratio_eq_1_0_steps"] == 1
        assert result["config"]["termination_token_ids"] == {"eos": [2], "pad": [0]}

    @pytest.mark.asyncio
    async def test_train_grpo_model_seeds_before_loading_model(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_grpo_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_order: list = []

        def grpo_config_init(self, **kwargs):
            pass

        fake_grpo_config = type("GRPOConfig", (), {"__init__": grpo_config_init})
        fake_grpo_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("num_generations", inspect.Parameter.KEYWORD_ONLY, default=4),
        ])

        def trainer_init(self, **kwargs):
            self.model = kwargs["model"]
            self.state = types.SimpleNamespace(log_history=[], global_step=0, epoch=0)

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.2)

        fake_grpo_trainer = type("GRPOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_grpo_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("reward_funcs", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.GRPOConfig = fake_grpo_config
        fake_trl.GRPOTrainer = fake_grpo_trainer

        def record_seed(seed):
            captured_order.append(("seed", seed))

        def record_load(*args, **kwargs):
            captured_order.append(("load", args[0]))
            return mock_model, mock_tokenizer

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_set_global_seed", side_effect=record_seed),
            patch.object(svc, "_load_model_and_tokenizer", side_effect=record_load),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
            patch.object(svc, "_apply_lora_to_model", return_value=(mock_model, None, 5)),
        ):
            result = await svc.train_grpo_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_grpo_seed_before_load"),
                seed=3407,
            )

        assert result["success"] is True
        assert captured_order[:2] == [
            ("seed", 3407),
            ("load", svc.config.base_model),
        ]

    @pytest.mark.asyncio
    async def test_train_grpo_model_can_continue_from_existing_lora_adapter(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_grpo_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        resumed_model = MagicMock()
        resumed_tokenizer = MagicMock()
        captured_trainer_kwargs: dict = {}

        def grpo_config_init(self, **kwargs):
            pass

        fake_grpo_config = type("GRPOConfig", (), {"__init__": grpo_config_init})
        fake_grpo_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("num_generations", inspect.Parameter.KEYWORD_ONLY, default=4),
            inspect.Parameter("max_prompt_length", inspect.Parameter.KEYWORD_ONLY, default=512),
            inspect.Parameter("max_completion_length", inspect.Parameter.KEYWORD_ONLY, default=256),
        ])

        def trainer_init(self, **kwargs):
            captured_trainer_kwargs.update(kwargs)
            self.model = kwargs["model"]
            self.state = types.SimpleNamespace(log_history=[], global_step=0, epoch=0)

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.2)

        fake_grpo_trainer = type("GRPOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_grpo_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("reward_funcs", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.GRPOConfig = fake_grpo_config
        fake_trl.GRPOTrainer = fake_grpo_trainer

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(
                svc,
                "_load_existing_lora_adapter",
                return_value=(resumed_model, resumed_tokenizer, 9),
            ) as load_existing,
        ):
            result = await svc.train_grpo_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_grpo_continue_adapter"),
                use_lora=True,
                adapter_path="/models/best_sft",
            )

        assert result["success"] is True
        load_existing.assert_called_once()
        assert captured_trainer_kwargs["model"] is resumed_model
        assert "peft_config" not in captured_trainer_kwargs
        assert result["config"]["continued_from_adapter"] is True
        assert result["config"]["adapter_path"] == "/models/best_sft"
        assert result["config"]["lora_trainable_fp32_tensors"] == 9

    @pytest.mark.asyncio
    async def test_train_grpo_model_rejects_quantized_full_finetuning_without_lora(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_grpo_data()

        result = await svc.train_grpo_model(
            dataset=dataset,
            output_dir=str(tmp_path / "test_grpo_quantized_full_model"),
            use_lora=False,
            load_in_4bit=True,
        )

        assert result["success"] is False
        assert "requires use_lora=True" in result["error"]

    @pytest.mark.asyncio
    async def test_train_kto_model_can_continue_from_existing_lora_adapter(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_kto_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        resumed_model = MagicMock()
        resumed_tokenizer = MagicMock()
        captured_trainer_kwargs: dict = {}

        def kto_config_init(self, **kwargs):
            pass

        fake_kto_config = type("KTOConfig", (), {"__init__": kto_config_init})
        fake_kto_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("beta", inspect.Parameter.KEYWORD_ONLY, default=0.1),
            inspect.Parameter("desirable_weight", inspect.Parameter.KEYWORD_ONLY, default=1.0),
            inspect.Parameter("undesirable_weight", inspect.Parameter.KEYWORD_ONLY, default=1.0),
        ])

        def trainer_init(self, **kwargs):
            captured_trainer_kwargs.update(kwargs)
            self.model = kwargs["model"]

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.15)

        fake_kto_trainer = type("KTOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_kto_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("callbacks", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("peft_config", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.KTOConfig = fake_kto_config
        fake_trl.KTOTrainer = fake_kto_trainer

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(
                svc,
                "_load_existing_lora_adapter",
                return_value=(resumed_model, resumed_tokenizer, 11),
            ) as load_existing,
        ):
            result = await svc.train_kto_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_kto_continue_adapter"),
                use_lora=True,
                adapter_path="/models/best_sft",
            )

        assert result["success"] is True
        load_existing.assert_called_once()
        assert captured_trainer_kwargs["model"] is resumed_model
        assert "peft_config" not in captured_trainer_kwargs
        assert result["config"]["continued_from_adapter"] is True
        assert result["config"]["adapter_path"] == "/models/best_sft"
        assert result["config"]["lora_trainable_fp32_tensors"] == 11

    @pytest.mark.asyncio
    async def test_train_kto_model_seeds_before_loading_model(self, tmp_path):
        svc = TrainingService()
        dataset = _sample_kto_data()
        mock_model, mock_tokenizer = _mock_model_and_tokenizer()
        captured_order: list = []

        def kto_config_init(self, **kwargs):
            pass

        fake_kto_config = type("KTOConfig", (), {"__init__": kto_config_init})
        fake_kto_config.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("output_dir", inspect.Parameter.KEYWORD_ONLY, default="."),
            inspect.Parameter("eval_strategy", inspect.Parameter.KEYWORD_ONLY, default="no"),
            inspect.Parameter("beta", inspect.Parameter.KEYWORD_ONLY, default=0.1),
            inspect.Parameter("desirable_weight", inspect.Parameter.KEYWORD_ONLY, default=1.0),
            inspect.Parameter("undesirable_weight", inspect.Parameter.KEYWORD_ONLY, default=1.0),
        ])

        def trainer_init(self, **kwargs):
            self.model = kwargs["model"]

        def trainer_train(self, **kwargs):
            return types.SimpleNamespace(training_loss=0.15)

        fake_kto_trainer = type("KTOTrainer", (), {
            "__init__": trainer_init,
            "train": trainer_train,
            "save_model": MagicMock(),
        })
        fake_kto_trainer.__init__.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("args", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("train_dataset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("processing_class", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])

        fake_trl = types.ModuleType("trl")
        fake_trl.KTOConfig = fake_kto_config
        fake_trl.KTOTrainer = fake_kto_trainer

        def record_seed(seed):
            captured_order.append(("seed", seed))

        def record_load(*args, **kwargs):
            captured_order.append(("load", args[0]))
            return mock_model, mock_tokenizer

        with (
            _intercept_imports({"trl": fake_trl}),
            patch.object(svc, "_set_global_seed", side_effect=record_seed),
            patch.object(svc, "_load_model_and_tokenizer", side_effect=record_load),
            patch.object(svc, "_detect_precision", return_value=(False, False)),
            patch.object(svc, "_build_lora_config", return_value=MagicMock()),
        ):
            result = await svc.train_kto_model(
                dataset=dataset,
                output_dir=str(tmp_path / "test_kto_seed_before_load"),
                seed=3407,
            )

        assert result["success"] is True
        assert captured_order[:2] == [
            ("seed", 3407),
            ("load", svc.config.base_model),
        ]
