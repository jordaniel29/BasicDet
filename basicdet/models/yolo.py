"""YOLO26 pipeline — train, evaluate, and predict for the YOLO family.

Thin orchestration over Ultralytics: owns seeding, device resolution, and W&B
wiring, then delegates the heavy lifting (training loop, architecture, losses,
data loading) to Ultralytics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from basicdet.utils import tracking
from basicdet.utils.config import YOLOExperimentConfig
from basicdet.utils.runtime import resolve_device
from basicdet.utils.seed import set_seed

logger = logging.getLogger(__name__)

# NMS IoU threshold used at inference time.
_PREDICT_IOU = 0.7


def train(config: YOLOExperimentConfig) -> Any:
    """Fine-tune a YOLO26 model on the dataset.

    Args:
        config: The validated experiment configuration.

    Returns:
        The Ultralytics training results object (metrics + saved-run paths).

    Raises:
        FileNotFoundError: If the dataset config YAML does not exist.
    """
    if not config.data.yaml_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config.data.yaml_path}")

    set_seed(config.train.seed, deterministic=config.train.deterministic)
    device = resolve_device(config.train.device)

    model = YOLO(config.model.weights)

    if tracking.resolve_wandb_enabled(config.wandb):
        tracking.init_wandb(config)
        tracking.register_callbacks(model, config)

    logger.info(
        "Starting YOLO26 fine-tune: weights=%s, data=%s, epochs=%d, device=%s",
        config.model.weights,
        config.data.yaml_path,
        config.train.epochs,
        device,
    )

    return model.train(
        data=str(config.data.yaml_path),
        imgsz=config.model.imgsz,
        epochs=config.train.epochs,
        batch=config.train.batch,
        optimizer=config.train.optimizer,
        lr0=config.train.lr0,
        patience=config.train.patience,
        workers=config.train.workers,
        fraction=config.train.fraction,
        cache=config.train.cache,
        seed=config.train.seed,
        deterministic=config.train.deterministic,
        device=device,
        project=config.train.project,
        name=config.train.name,
        # Last so an `extra` key deliberately overrides the typed field above.
        **config.train.extra,
    )


def evaluate(
    config: YOLOExperimentConfig,
    weights: str,
    split: str = "test",
    conf: float | None = None,
) -> Any:
    """Evaluate trained weights on a dataset split (mAP, precision, recall).

    Args:
        config: The experiment configuration (provides dataset + run settings).
        weights: Path to the checkpoint to evaluate (e.g. ``best.pt``).
        split: Which split to evaluate — ``"train"``, ``"val"``, or ``"test"``.
        conf: Confidence threshold. ``None`` uses Ultralytics' low default
            (~0.001), which is correct for mAP; raise it only for thresholded
            precision/recall.

    Returns:
        The Ultralytics validation metrics object.

    Raises:
        FileNotFoundError: If the dataset config YAML does not exist.
    """
    if not config.data.yaml_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config.data.yaml_path}")

    device = resolve_device(config.train.device)
    model = YOLO(weights)

    kwargs: dict[str, Any] = {}
    if conf is not None:
        kwargs["conf"] = conf

    logger.info("Evaluating %s on split=%s (device=%s)", weights, split, device)
    return model.val(
        data=str(config.data.yaml_path),
        imgsz=config.model.imgsz,
        split=split,
        device=device,
        project=config.train.project,
        name=f"{config.train.name}_eval_{split}",
        **kwargs,
    )


def predict(
    config: YOLOExperimentConfig,
    weights: str,
    source: Path,
    output: Path,
    conf: float = 0.25,
) -> Any:
    """Run inference and save annotated outputs to ``output``.

    Args:
        config: The experiment configuration (provides imgsz + device).
        weights: Path to the checkpoint (e.g. ``best.pt``).
        source: Image/video path, directory, or glob.
        output: Directory for annotated outputs (Ultralytics writes to
            ``runs/detect/<output.parent>/<output.name>``).
        conf: Confidence threshold for displayed detections.

    Returns:
        The list of Ultralytics ``Results``.

    Raises:
        FileNotFoundError: If ``weights`` does not exist.
    """
    if not Path(weights).is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out = Path(output)
    device = resolve_device(config.train.device)
    model = YOLO(weights)

    logger.info("Running inference: weights=%s, source=%s, conf=%.2f", weights, source, conf)
    results = model.predict(
        source=str(source),
        conf=conf,
        iou=_PREDICT_IOU,
        imgsz=config.model.imgsz,
        device=device,
        save=True,
        project=str(out.parent),
        name=out.name,
    )

    total = sum(len(r.boxes) for r in results if r.boxes is not None)
    logger.info("Processed %d item(s), %d detection(s); outputs in %s", len(results), total, out)
    return results
