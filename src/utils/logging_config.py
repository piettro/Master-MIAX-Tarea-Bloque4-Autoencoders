"""Root-logger configuration (called once from ``main.py``)."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
# Third-party loggers that are too chatty at INFO level.
NOISY_LOGGERS = ("matplotlib", "PIL", "urllib3")
# TensorFlow also prints its own copy of warnings; keep only real errors.
TF_LOGGERS = ("tensorflow", "absl")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger with a single console handler.

    Args:
        level: Logging level for the project loggers (name or number).
    """
    logging.basicConfig(
        level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in TF_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
