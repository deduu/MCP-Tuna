from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from finetuning_pipeline.services.inference_service import InferenceService


def test_load_text_tokenizer_uses_left_padding_and_pad_fallback():
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    tokenizer.padding_side = "right"

    with patch(
        "finetuning_pipeline.services.inference_service.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ):
        loaded = InferenceService._load_text_tokenizer("test/model")

    assert loaded is tokenizer
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.padding_side == "left"


def test_build_text_quantization_config_matches_notebook_fp16():
    config = InferenceService._build_text_quantization_config("4bit")

    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_compute_dtype == torch.float16


def test_load_text_model_and_tokenizer_sets_eval_and_fp16():
    svc = InferenceService()
    tokenizer = MagicMock()
    model = MagicMock()
    model.eval = MagicMock()

    with (
        patch.object(svc, "_load_text_tokenizer", return_value=tokenizer),
        patch(
            "finetuning_pipeline.services.inference_service.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as mock_from_pretrained,
    ):
        loaded_model, loaded_tokenizer = svc._load_text_model_and_tokenizer(
            model_path="test/model",
            adapter_path=None,
            quantization="4bit",
        )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    model.eval.assert_called_once_with()
    assert mock_from_pretrained.call_args.kwargs["torch_dtype"] == torch.float16
