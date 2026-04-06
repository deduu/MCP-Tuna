from __future__ import annotations

from typing import Optional


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    seed = int(seed)

    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:
        from transformers import set_seed

        set_seed(seed)
    except Exception:
        pass
