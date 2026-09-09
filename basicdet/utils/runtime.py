"""Device selection and run-provenance helpers."""

from __future__ import annotations

import logging
import subprocess

import torch

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """Resolve a device spec to a concrete Ultralytics device string.

    Args:
        device: ``"auto"``, ``"cpu"``, or a CUDA index string (``"0"``, ``"0,1"``).

    Returns:
        ``"cpu"`` or a CUDA index string. ``"auto"`` resolves to ``"0"`` when a
        CUDA device is available, otherwise ``"cpu"``.
    """
    if device != "auto":
        return device
    resolved = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Auto-selected device: %s", resolved)
    return resolved


def get_git_commit() -> str:
    """Return the current git commit SHA, or ``"unknown"`` if unavailable.

    Used to tie every tracked run back to the exact code that produced it.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def resolve_torch_device(device: str) -> str:
    """Resolve a device spec to a concrete torch device string.

    The native ReID trainers talk to torch directly, so they need
    ``"cuda:N"`` rather than the bare index string Ultralytics wants (see
    :func:`resolve_device`).

    Args:
        device: ``"auto"``, ``"cpu"``, a CUDA index string (``"1"``), or an
            explicit torch device (``"cuda:1"``).

    Returns:
        ``"cpu"`` or ``"cuda:N"``. ``"auto"`` resolves to ``"cuda:0"`` when a
        CUDA device is available, otherwise ``"cpu"``.
    """
    if device == "auto":
        resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Auto-selected torch device: %s", resolved)
        return resolved
    if device.isdigit():
        return f"cuda:{device}"
    return device
