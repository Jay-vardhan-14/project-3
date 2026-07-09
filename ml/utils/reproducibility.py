"""Reproducibility helpers for training and data preparation."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int, include_torch: bool = True) -> None:
    """Set all available random seeds for deterministic training runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not include_torch:
        return

    try:
        import torch
    except Exception as exc:
        LOGGER.warning("PyTorch seed setup skipped: %s", exc)
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
