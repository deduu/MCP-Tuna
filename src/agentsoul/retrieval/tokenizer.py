"""
Simple text tokenizer for BM25 indexing and search.

Splits text into lowercase alphanumeric tokens.
"""

import re
from typing import List

_WORD_RE = re.compile(r"[0-9A-Za-z_]+", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words.

    Args:
        text: The input text to tokenize.

    Returns:
        List of lowercase tokens. Empty list if input is not a string.
    """
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())
