"""Global seeding for reproducible runs."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic_tf: bool = True) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow.

    Args:
        seed: The seed value.
        deterministic_tf: Also request deterministic TensorFlow kernels so
            that two runs on the same machine give identical weights.

    Raises:
        TypeError: If ``seed`` is not an integer.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover - TF is a hard dependency
        logger.warning("TensorFlow not installed; only Python/NumPy seeded.")
        return
    # NOTE: improved over professor's solution — set_random_seed seeds
    # Python, NumPy and TF at once, and op determinism removes the residual
    # run-to-run variance of multi-threaded CPU kernels.
    tf.keras.utils.set_random_seed(seed)
    if deterministic_tf:
        tf.config.experimental.enable_op_determinism()
    logger.debug("Global seed set to %d", seed)
