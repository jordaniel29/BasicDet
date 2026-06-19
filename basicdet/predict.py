"""Entrypoint: run a trained model and save annotated predictions.

The config's ``family`` field selects the pipeline.

Usage:
    python basicdet/predict.py --config configs/yolo26_person.yaml \
        --weights runs/detect/yolo26/basicdet/weights/best.pt \
        --source assets/data/persondet_v1.1/images/test \
        --output runs/predict/basicdet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from basicdet.utils import registry
from basicdet.utils.config import load_experiment
from basicdet.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection inference and save outputs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config.")
    parser.add_argument("--weights", required=True, help="Path to the checkpoint.")
    parser.add_argument("--source", type=Path, required=True, help="Image/video path or directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/predict/basicdet"),
        help="Directory for annotated outputs.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_experiment(args.config)
    registry.predict(
        config, weights=args.weights, source=args.source, output=args.output, conf=args.conf
    )


if __name__ == "__main__":
    main()
