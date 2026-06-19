"""Logging configuration for entrypoint scripts."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, at the start of an entrypoint.

    Args:
        level: Minimum log level to emit (e.g. ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
