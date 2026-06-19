"""RF-DETR pipeline — train, evaluate, and predict for the RF-DETR family.

Thin orchestration over the ``rfdetr`` package: owns seeding and W&B wiring,
then delegates training to RF-DETR (which auto-discovers ``train/``/``valid/``/
``test/``, each with an ``_annotations.coco.json``, under ``dataset_dir``).

Experiment tracking: RF-DETR has native W&B support, so we pass its ``wandb``/
``project``/``run`` flags through. W&B automatically captures the git commit of
the working tree, and RF-DETR logs its training args as the run config — giving
the same run -> config + commit traceability as the YOLO path (see CLAUDE.md).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase, RFDETRLarge

from basicdet.metrics.coco import PredictFn, evaluate_coco
from basicdet.utils.config import RFDETRExperimentConfig
from basicdet.utils.seed import set_seed

logger = logging.getLogger(__name__)

# RF-DETR uses "valid" (not "val") for its COCO validation subfolder.
_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}

# COCO mAP needs a low confidence floor so recall isn't truncated.
_DEFAULT_EVAL_CONF = 0.05

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(config: RFDETRExperimentConfig) -> Any:
    """Construct an (untrained) RF-DETR model from config.

    Args:
        config: The experiment configuration.

    Returns:
        An ``RFDETRBase`` or ``RFDETRLarge`` instance.
    """
    kwargs: dict[str, Any] = {}
    if config.model.resolution is not None:
        kwargs["resolution"] = config.model.resolution
    if config.model.num_classes is not None:
        kwargs["num_classes"] = config.model.num_classes

    model_cls = RFDETRLarge if config.model.variant == "large" else RFDETRBase
    logger.info("Building RF-DETR (%s) with %s", config.model.variant, kwargs or "defaults")
    return model_cls(**kwargs)


def load_model(config: RFDETRExperimentConfig, weights: str) -> Any:
    """Construct the configured RF-DETR variant from a trained checkpoint.

    Args:
        config: The experiment configuration (provides variant + resolution).
        weights: Path to the trained checkpoint (``.pth``).

    Returns:
        An RF-DETR model with the trained weights loaded, ready for inference.
    """
    kwargs: dict[str, Any] = {"pretrain_weights": weights}
    if config.model.resolution is not None:
        kwargs["resolution"] = config.model.resolution
    model_cls = RFDETRLarge if config.model.variant == "large" else RFDETRBase
    return model_cls(**kwargs)


def _build_predict_fn(model: Any, conf: float) -> PredictFn:
    """Wrap an RF-DETR model as a ``PredictFn`` for COCO evaluation.

    Args:
        model: A loaded RF-DETR model exposing ``predict``.
        conf: Confidence threshold below which detections are dropped.

    Returns:
        A callable mapping an image path to ``(boxes_xyxy, scores, class_ids)``.
    """

    def _predict(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=conf)
        return detections.xyxy, detections.confidence, detections.class_id

    return _predict


def train(config: RFDETRExperimentConfig) -> None:
    """Fine-tune an RF-DETR model on the dataset.

    Args:
        config: The validated experiment configuration.

    Raises:
        FileNotFoundError: If ``dataset_dir`` does not exist.
    """
    if not config.data.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {config.data.dataset_dir}")

    set_seed(config.train.seed)

    # RF-DETR reads the W&B entity from the environment rather than a kwarg.
    if config.wandb.enabled and config.wandb.entity:
        os.environ.setdefault("WANDB_ENTITY", config.wandb.entity)

    model = build_model(config)

    logger.info(
        "Starting RF-DETR fine-tune: variant=%s, data=%s, epochs=%d",
        config.model.variant,
        config.data.dataset_dir,
        config.train.epochs,
    )

    model.train(
        dataset_dir=str(config.data.dataset_dir),
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        grad_accum_steps=config.train.grad_accum_steps,
        lr=config.train.lr,
        num_workers=config.train.num_workers,
        early_stopping=config.train.early_stopping,
        output_dir=config.train.output_dir,
        tensorboard=config.train.tensorboard,
        wandb=config.wandb.enabled,
        project=config.wandb.project,
        run=config.train.run_name,
        **config.train.extra,
    )
    logger.info("Training finished. Checkpoints in %s", config.train.output_dir)


def evaluate(
    config: RFDETRExperimentConfig,
    weights: str,
    split: str = "test",
    conf: float | None = None,
) -> dict[str, float]:
    """Evaluate trained RF-DETR weights on a split with COCO metrics.

    Args:
        config: The experiment configuration (provides dataset + model variant).
        weights: Path to the trained checkpoint (e.g. ``checkpoint_best_ema.pth``).
        split: ``"train"``, ``"val"``, or ``"test"``.
        conf: Confidence floor for detections fed to the evaluator. ``None`` uses
            a low default (0.05) appropriate for mAP.

    Returns:
        Mapping of COCO metric name to value (mAP_50_95, mAP_50, ...).

    Raises:
        FileNotFoundError: If the split directory or its annotations are missing.
        KeyError: If ``split`` is not one of train/val/test.
    """
    split_dir = config.data.dataset_dir / _SPLIT_DIRS[split]
    annotations_json = split_dir / "_annotations.coco.json"
    effective_conf = _DEFAULT_EVAL_CONF if conf is None else conf

    model = load_model(config, weights)

    logger.info("Evaluating RF-DETR weights=%s on split=%s", weights, split)
    metrics = evaluate_coco(
        _build_predict_fn(model, effective_conf),
        images_dir=split_dir,
        annotations_json=annotations_json,
    )
    logger.info("RF-DETR %s metrics: %s", split, metrics)
    return metrics


def predict(
    config: RFDETRExperimentConfig,
    weights: str,
    source: Path,
    output: Path,
    conf: float = 0.5,
) -> int:
    """Run inference on an image or directory and save annotated outputs.

    Args:
        config: The experiment configuration (provides the model variant).
        weights: Path to the trained checkpoint (``.pth``).
        source: An image file or directory of images.
        output: Directory for annotated outputs (created if absent).
        conf: Confidence threshold for displayed detections.

    Returns:
        Total number of person detections across processed images.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
    """
    source, output = Path(source), Path(output)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    model = load_model(config, weights)
    image_paths = (
        sorted(p for p in source.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        if source.is_dir()
        else [source]
    )

    output.mkdir(parents=True, exist_ok=True)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    total_detections = 0
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=conf)
        labels = [f"person {score:.2f}" for score in detections.confidence]

        annotated = box_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)
        annotated.save(output / image_path.name)
        total_detections += len(detections)

    logger.info(
        "Annotated %d image(s), %d detection(s); outputs in %s",
        len(image_paths),
        total_detections,
        output,
    )
    return total_detections
