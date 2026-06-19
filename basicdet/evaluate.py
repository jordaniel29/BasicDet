"""Entrypoint: evaluate trained weights on a dataset split.

The config's ``family`` field selects the pipeline (YOLO -> Ultralytics mAP,
RF-DETR -> COCO mAP).

Usage:
    python basicdet/evaluate.py --config configs/yolo26_person.yaml \
        --weights runs/detect/yolo26/basicdet/weights/best.pt --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from basicdet.utils import registry
from basicdet.utils.config import load_experiment
from basicdet.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection weights.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config.")
    parser.add_argument("--weights", required=True, help="Path to the checkpoint.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold; omit for the family's mAP-appropriate default.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_experiment(args.config)
    registry.evaluate(config, weights=args.weights, split=args.split, conf=args.conf)


if __name__ == "__main__":
    main()
