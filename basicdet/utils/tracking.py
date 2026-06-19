"""Weights & Biases integration for Ultralytics training.

Ultralytics ships its own (optional) W&B logger, but we drive W&B explicitly so
that every run records the full experiment config and the exact git commit —
the traceability requirement in CLAUDE.md. Ultralytics' built-in integration is
disabled here to avoid a duplicate run being created.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO, settings

from basicdet.utils.config import YOLOExperimentConfig
from basicdet.utils.runtime import get_git_commit

logger = logging.getLogger(__name__)


def init_wandb(config: YOLOExperimentConfig) -> None:
    """Start a W&B run logging the full config and git commit.

    Args:
        config: The experiment config. Serialized in full to ``wandb.config``.
    """
    # Imported lazily so the package is only required when tracking is enabled.
    import wandb

    # Disable Ultralytics' own W&B hook so we don't open two runs.
    settings.update({"wandb": False})

    git_commit = get_git_commit()
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.train.name,
        config={**config.model_dump(mode="json"), "git_commit": git_commit},
        tags=[f"git:{git_commit[:8]}", config.model.weights],
    )
    logger.info("W&B run initialized (project=%s, commit=%s)", config.wandb.project, git_commit[:8])


def register_callbacks(model: YOLO, config: YOLOExperimentConfig) -> None:
    """Attach W&B logging callbacks to an Ultralytics model.

    Logs per-epoch validation metrics, and optionally uploads the best
    checkpoint as a model artifact when training finishes.

    Args:
        model: The Ultralytics model to instrument.
        config: The experiment config (controls model-artifact upload).
    """
    # Imported lazily so the package is only required when tracking is enabled.
    import wandb

    def _on_fit_epoch_end(trainer: object) -> None:
        # trainer.metrics: dict of val metrics (e.g. "metrics/mAP50(B)").
        metrics = getattr(trainer, "metrics", None) or {}
        epoch = getattr(trainer, "epoch", None)
        wandb.log({**metrics, "epoch": epoch})

    def _on_train_end(trainer: object) -> None:
        if config.wandb.log_model:
            best = getattr(trainer, "best", None)
            if best and Path(best).is_file():
                artifact = wandb.Artifact(name=f"{config.train.name}-weights", type="model")
                artifact.add_file(str(best))
                wandb.log_artifact(artifact)
                logger.info("Logged best checkpoint to W&B: %s", best)
        wandb.finish()

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_train_end)
