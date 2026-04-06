from __future__ import annotations

from shared.model_load_errors import format_model_load_error, is_windows_pagefile_error


def test_detects_windows_pagefile_error_markers():
    exc = OSError("The paging file is too small for this operation to complete. (os error 1455)")
    assert is_windows_pagefile_error(exc) is True


def test_formats_actionable_pagefile_message():
    exc = OSError("The paging file is too small for this operation to complete. (os error 1455)")
    message = format_model_load_error(
        exc,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        load_in_4bit=True,
    )

    assert "virtual memory/paging file is too small" in message
    assert "Increase the paging file" in message
    assert "4-bit quantization reduces GPU memory after load" in message
