from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import pytest

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


def test_run_loaded_text_messages_inference_passes_explicit_thinking_flag():
    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "prompt"
    model_inputs = MagicMock()
    model_inputs.input_ids = torch.tensor([[1, 2, 3]])
    model_inputs.to.return_value = model_inputs
    tokenizer.return_value = model_inputs
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    tokenizer.decode.return_value = "final answer"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    result = InferenceService._run_loaded_text_messages_inference(
        model=model,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "Hi"}],
        max_new_tokens=32,
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        do_sample=False,
        model_path="test/model",
        adapter_path=None,
        quantization="4bit",
        thinking_mode="off",
    )

    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert result["success"] is True
    assert result["thinking_mode"] == "off"


@pytest.mark.asyncio
async def test_run_inference_reuses_loaded_model_for_all_prompts():
    svc = InferenceService()
    model = MagicMock()
    tokenizer = MagicMock()

    with (
        patch.object(
            svc,
            "_load_text_model_and_tokenizer",
            return_value=(model, tokenizer),
        ) as mock_load,
        patch.object(
            svc,
            "_run_loaded_text_messages_inference",
            side_effect=[
                {
                    "success": True,
                    "response": "resp-1",
                    "generation_time_seconds": 1.0,
                    "tokens_generated": 10,
                    "tokens_per_second": 10.0,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                },
                {
                    "success": True,
                    "response": "resp-2",
                    "generation_time_seconds": 2.0,
                    "tokens_generated": 20,
                    "tokens_per_second": 10.0,
                    "usage": {"prompt_tokens": 6, "completion_tokens": 20, "total_tokens": 26},
                },
            ],
        ) as mock_run,
        patch.object(svc.gpu, "clear_gpu_memory") as mock_clear,
    ):
        result = await svc.run_inference(
            prompts=["first", "second"],
            model_path="test/model",
            adapter_path="test/adapter",
            max_new_tokens=64,
            thinking_mode="on",
        )

    assert result["success"] is True
    assert [entry["response"] for entry in result["results"]] == ["resp-1", "resp-2"]
    mock_load.assert_called_once_with(
        model_path="test/model",
        adapter_path="test/adapter",
        quantization="4bit",
    )
    assert mock_run.call_count == 2
    assert all(call.kwargs["thinking_mode"] == "on" for call in mock_run.call_args_list)
    mock_clear.assert_called_once_with()
