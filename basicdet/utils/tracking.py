"""Weights & Biases integration and availability checks.

Ultralytics ships its own (optional) W&B logger, but we drive W&B explicitly so
that every run records the full experiment config and the exact git commit —
the traceability requirement in CLAUDE.md. Ultralytics' built-in integration is
disabled here to avoid a duplicate run being created.

``resolve_wandb_enabled()`` is model-agnostic and is the gate every family should use
instead of reading ``config.wandb.enabled`` directly.
"""

from __future__ import annotations

import logging
import netrc
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from basicdet.utils.config import WandbConfig, YOLOExperimentConfig
from basicdet.utils.runtime import get_git_commit

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# W&B modes that only ever write to disk, so they need no account.
_LOCAL_ONLY_MODES = frozenset({"offline", "disabled", "dryrun"})

# Host that `wandb login` writes into ~/.netrc for W&B Cloud.
_DEFAULT_WANDB_HOST = "api.wandb.ai"


def _wandb_host() -> str:
    """Return the netrc host to look for, honouring self-hosted ``WANDB_BASE_URL``."""
    base_url = os.environ.get("WANDB_BASE_URL")
    if not base_url:
        return _DEFAULT_WANDB_HOST
    return urlparse(base_url).hostname or _DEFAULT_WANDB_HOST


def _has_credentials() -> bool:
    """Report whether W&B credentials exist, without triggering a login prompt.

    Deliberately inspects the environment and ``~/.netrc`` by hand rather than
    calling into ``wandb``: the library's own resolution path can block on an
    interactive prompt, which is exactly what this check exists to avoid.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return _wandb_host() in netrc.netrc().hosts
    except (OSError, netrc.NetrcParseError):
        return False


def resolve_wandb_enabled(config: WandbConfig) -> bool:
    """Decide whether W&B logging should run for this experiment.

    Tracking is opt-out via ``wandb.enabled``, but an *enabled* run on a machine
    with no credentials would otherwise stall on ``wandb``'s interactive login
    prompt — fatal for the unattended tmux/nohup runs this repo relies on (a
    multi-hour training would hang at step 0 with no indication why). Missing
    credentials are therefore treated as "tracking off, keep training".

    Args:
        config: The experiment's W&B settings.

    Returns:
        True if W&B should be initialized.
    """
    if not config.enabled:
        return False

    if os.environ.get("WANDB_MODE", "").lower() in _LOCAL_ONLY_MODES:
        return True

    if _has_credentials():
        return True

    logger.warning(
        "W&B is enabled in the config but no credentials were found — continuing "
        "WITHOUT experiment tracking. To enable it, either run `wandb login`, set "
        "WANDB_API_KEY, or set WANDB_MODE=offline to log locally. To silence this, "
        "set wandb.enabled: false in the config."
    )
    return False


def init_wandb(config: YOLOExperimentConfig) -> None:
    """Start a W&B run logging the full config and git commit.

    Args:
        config: The experiment config. Serialized in full to ``wandb.config``.
    """
    # Imported lazily so neither package is required when tracking is disabled,
    # and so the RF-DETR path can use resolve_wandb_enabled() without pulling Ultralytics.
    import wandb
    from ultralytics import settings

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
