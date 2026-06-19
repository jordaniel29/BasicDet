"""COCO-style mAP evaluation for any detector.

Decoupled from the model: callers pass a ``predict_fn`` that maps an image path
to detected boxes. This keeps the metric code reusable across RF-DETR, YOLO, or
anything else, and makes the YOLO/RF-DETR numbers directly comparable since both
are scored against the same COCO ground truth with the same protocol.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger(__name__)

# A detector: image path -> (boxes_xyxy [N,4], scores [N], class_ids [N]).
PredictFn = Callable[[Path], tuple[np.ndarray, np.ndarray, np.ndarray]]

# Index of each summary metric within COCOeval.stats (bbox eval order).
_STAT_KEYS = [
    "mAP_50_95",
    "mAP_50",
    "mAP_75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def evaluate_coco(
    predict_fn: PredictFn,
    images_dir: Path,
    annotations_json: Path,
) -> dict[str, float]:
    """Score a detector against COCO ground truth and return summary metrics.

    Args:
        predict_fn: Maps an image path to ``(boxes_xyxy, scores, class_ids)``,
            where boxes are absolute-pixel ``[x1, y1, x2, y2]``.
        images_dir: Directory holding the images named in the annotations.
        annotations_json: COCO ground-truth annotations file.

    Returns:
        Mapping of metric name (e.g. ``"mAP_50_95"``, ``"mAP_50"``) to value.
        All-zero metrics are returned if the model produced no detections.

    Raises:
        FileNotFoundError: If ``images_dir`` or ``annotations_json`` is missing.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_json.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_json}")

    coco_gt = COCO(str(annotations_json))
    cat_ids = coco_gt.getCatIds()
    # Single-class dataset: every detection maps to the lone category id.
    single_category_id = cat_ids[0] if len(cat_ids) == 1 else None

    results: list[dict[str, object]] = []
    for image in coco_gt.dataset["images"]:
        boxes, scores, class_ids = predict_fn(images_dir / image["file_name"])
        for (x1, y1, x2, y2), score, class_id in zip(boxes, scores, class_ids, strict=True):
            results.append(
                {
                    "image_id": image["id"],
                    "category_id": single_category_id or int(class_id) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    if not results:
        logger.warning("Model produced no detections; returning zero metrics.")
        return dict.fromkeys(_STAT_KEYS, 0.0)

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {key: float(value) for key, value in zip(_STAT_KEYS, coco_eval.stats, strict=True)}
