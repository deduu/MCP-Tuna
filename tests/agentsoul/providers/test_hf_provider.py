import logging
from unittest.mock import Mock

from agentsoul.providers.hf import _load_tokenizer_with_fallback


def test_load_tokenizer_with_fallback_handles_llama_fast_tokenizer_none_vocab_error():
    auto_tokenizer = Mock()
    slow_tokenizer = object()
    auto_tokenizer.from_pretrained.side_effect = [
        AttributeError("'NoneType' object has no attribute 'endswith'"),
        slow_tokenizer,
    ]

    result = _load_tokenizer_with_fallback(
        auto_tokenizer,
        "test/model",
        logging.getLogger("test"),
        trust_remote_code=True,
    )

    assert result is slow_tokenizer
    assert auto_tokenizer.from_pretrained.call_count == 2
    first_call = auto_tokenizer.from_pretrained.call_args_list[0]
    second_call = auto_tokenizer.from_pretrained.call_args_list[1]
    assert first_call.kwargs["use_fast"] is True
    assert second_call.kwargs["use_fast"] is False
