"""Reproducibility helpers."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs that affect training for reproducible runs.

    Seeds Python's ``random``, NumPy, and Torch (CPU + all CUDA devices). When
    ``deterministic`` is set, also forces cuDNN into deterministic mode — this
    removes run-to-run variance at some throughput cost.

    Args:
        seed: The seed value.
        deterministic: If ``True``, disable cuDNN autotuning and request
            deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # PYTHONHASHSEED affects hash-based ordering in child dataloader procs.
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Seeded RNGs with seed=%d (deterministic=%s)", seed, deterministic)
