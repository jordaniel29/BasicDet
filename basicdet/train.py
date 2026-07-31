"""Entrypoint: fine-tune a detection model from a YAML experiment config.

The config's ``family`` field (``yolo`` or ``rfdetr``) selects the pipeline.

Usage:
    python basicdet/train.py --config configs/yolo26_person.yaml
    python basicdet/train.py --config configs/rfdetr_person.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from basicdet.utils import registry
from basicdet.utils.config import load_experiment
from basicdet.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a detection model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B for this run, overriding wandb.enabled in the config.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_experiment(args.config)
    if args.no_wandb:
        config.wandb.enabled = False
    registry.train(config)


if __name__ == "__main__":
    main()
