from __future__ import annotations


def is_windows_pagefile_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "paging file is too small",
            "os error 1455",
            "winerror 1455",
            "virtual memory is too small",
            "memory allocation of",
        )
    )


def format_model_load_error(
    exc: BaseException,
    *,
    model_name: str,
    load_in_4bit: bool,
) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    if is_windows_pagefile_error(exc):
        quantization_note = (
            "4-bit quantization reduces GPU memory after load, but Windows still has "
            "to map the base checkpoint weights first."
            if load_in_4bit
            else "The checkpoint still needs enough Windows virtual memory to be mapped before training starts."
        )
        return (
            f"Failed to load model '{model_name}': Windows virtual memory/paging file is too small "
            f"to map the checkpoint weights ({detail}). Increase the paging file, restart to free "
            f"commit space, or use a smaller or more heavily sharded base model. {quantization_note}"
        )
    return detail
