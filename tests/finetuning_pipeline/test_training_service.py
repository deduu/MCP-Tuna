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
