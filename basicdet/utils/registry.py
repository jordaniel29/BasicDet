"""Dispatch a loaded config to the matching model pipeline by its ``family``.

Imports are lazy and per-family, so running a YOLO job never imports RF-DETR
(and its heavy ``rfdetr``/``supervision``/``pycocotools`` stack), and vice
versa. Adding a new model family means adding one ``elif`` branch here plus a
module under ``basicdet/models/`` exposing ``train`` / ``evaluate`` / ``predict``.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from basicdet.utils.config import RFDETRExperimentConfig, YOLOExperimentConfig

AnyConfig = YOLOExperimentConfig | RFDETRExperimentConfig


def _model(family: str) -> ModuleType:
    """Return the pipeline module for a model family (imported lazily)."""
    if family == "yolo":
        from basicdet.models import yolo

        return yolo
    if family == "rfdetr":
        from basicdet.models import rfdetr

        return rfdetr
    raise ValueError(f"Unknown model family: {family!r}")


def train(config: AnyConfig) -> Any:
    """Run training for the config's model family."""
    return _model(config.family).train(config)


def evaluate(
    config: AnyConfig, weights: str, split: str = "test", conf: float | None = None
) -> Any:
    """Run evaluation for the config's model family."""
    return _model(config.family).evaluate(config, weights=weights, split=split, conf=conf)


def predict(config: AnyConfig, weights: str, source: Path, output: Path, conf: float = 0.25) -> Any:
    """Run inference for the config's model family."""
    return _model(config.family).predict(
        config, weights=weights, source=source, output=output, conf=conf
    )
