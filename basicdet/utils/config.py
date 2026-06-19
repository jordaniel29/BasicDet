"""Typed experiment configuration loaded from YAML.

Keeps all tunable values out of the Python source (see CLAUDE.md: configuration
as code). A single YAML file fully describes a training run; nothing in the
training code is hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, TypeAdapter

_ConfigT = TypeVar("_ConfigT", bound=BaseModel)


class DataConfig(BaseModel):
    """Dataset location.

    Attributes:
        yaml_path: Path to the Ultralytics dataset config (``data.yaml``). It
            defines the class names and the train/val/test image directories.
    """

    yaml_path: Path


class ModelConfig(BaseModel):
    """Model and input-resolution settings.

    Attributes:
        weights: Pretrained checkpoint to fine-tune from. YOLO26 variants:
            ``yolo26n.pt`` (nano) .. ``yolo26x.pt`` (xlarge); larger = more
            accurate and slower.
        imgsz: Square input resolution in pixels.
    """

    weights: str = "yolo26s.pt"
    imgsz: int = 640


class TrainConfig(BaseModel):
    """Training-loop hyperparameters passed through to Ultralytics.

    Attributes:
        epochs: Number of training epochs.
        batch: Images per batch. Use a float in (0, 1] to auto-size to a
            fraction of GPU memory (Ultralytics convention).
        optimizer: Optimizer name, or ``"auto"`` to let Ultralytics choose.
        lr0: Initial learning rate.
        patience: Early-stopping patience (epochs without val improvement).
        workers: Dataloader worker processes.
        seed: Global RNG seed for reproducibility.
        device: ``"auto"`` (pick CUDA if available), ``"cpu"``, or a CUDA index
            string such as ``"0"`` or ``"0,1"``.
        fraction: Fraction of the train set to use, in (0, 1]. ``1.0`` is the
            full dataset; lower values give fast smoke/debug runs.
        deterministic: Force deterministic algorithms (reproducible but slower).
        project: Ultralytics project label. Runs are written to
            ``runs/detect/<project>/<name>`` (Ultralytics prepends its
            ``runs_dir/detect``), so use a plain label, not a path.
        name: Run name; the run is written to ``<project>/<name>``.
    """

    epochs: int = 100
    batch: int | float = 16
    optimizer: str = "auto"
    lr0: float = 0.01
    patience: int = 50
    workers: int = 8
    seed: int = 42
    device: str = "auto"
    fraction: float = 1.0
    deterministic: bool = True
    project: str = "yolo26"
    name: str = "basicdet"


class WandbConfig(BaseModel):
    """Weights & Biases experiment-tracking settings.

    Attributes:
        enabled: Toggle W&B logging off for quick local debugging.
        project: W&B project name.
        entity: W&B team/user, or ``None`` for the logged-in default.
        log_model: Upload the best checkpoint as a W&B artifact at train end.
    """

    enabled: bool = True
    project: str = "person-det"
    entity: str | None = None
    log_model: bool = True


class YOLOExperimentConfig(BaseModel):
    """Full configuration for a single YOLO training/evaluation run.

    Attributes:
        family: Discriminator selecting the YOLO pipeline. Always ``"yolo"``.
    """

    family: Literal["yolo"] = "yolo"
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


# --------------------------------------------------------------------------- #
# RF-DETR
#
# RF-DETR consumes a COCO-layout directory (train/ valid/ test/, each with an
# _annotations.coco.json), not a YOLO data.yaml. Its train() API is also less
# standardized across releases than Ultralytics', so model construction and
# training take typed common fields plus an ``extra`` passthrough dict for
# version-specific kwargs — no Python edits needed to add an argument.
# --------------------------------------------------------------------------- #


class RFDETRDataConfig(BaseModel):
    """Dataset location for RF-DETR.

    Attributes:
        dataset_dir: Directory containing ``train/``, ``valid/``, ``test/``
            subfolders, each with images and an ``_annotations.coco.json``.
    """

    dataset_dir: Path


class RFDETRModelConfig(BaseModel):
    """Model variant and construction settings.

    Attributes:
        variant: ``"base"`` (RFDETRBase) or ``"large"`` (RFDETRLarge).
        resolution: Square input resolution in pixels; must be divisible by 56.
            ``None`` uses the variant's default.
        num_classes: Number of classes. ``None`` lets RF-DETR infer from the
            dataset (correct for this single-class ``person`` set).
    """

    variant: Literal["base", "large"] = "base"
    resolution: int | None = None
    num_classes: int | None = None


class RFDETRTrainConfig(BaseModel):
    """RF-DETR training hyperparameters.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Images per batch (RF-DETR is memory-hungry; 4 is typical).
        grad_accum_steps: Gradient-accumulation steps; effective batch =
            ``batch_size * grad_accum_steps``.
        lr: Base learning rate.
        num_workers: Dataloader worker processes.
        seed: Global RNG seed for reproducibility.
        early_stopping: Stop early when validation stops improving.
        tensorboard: Enable RF-DETR's TensorBoard logging.
        output_dir: Where checkpoints and logs are written.
        run_name: Run name (W&B run + output subdirectory label).
        extra: Extra keyword args forwarded verbatim to ``model.train`` — use
            for release-specific options without editing code.
    """

    epochs: int = 50
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    early_stopping: bool = True
    tensorboard: bool = False
    output_dir: str = "runs/rfdetr/basicdet"
    run_name: str = "basicdet"
    extra: dict[str, Any] = Field(default_factory=dict)


class RFDETRExperimentConfig(BaseModel):
    """Full configuration for an RF-DETR training/evaluation run.

    Attributes:
        family: Discriminator selecting the RF-DETR pipeline. Always ``"rfdetr"``.
    """

    family: Literal["rfdetr"] = "rfdetr"
    data: RFDETRDataConfig
    model: RFDETRModelConfig = Field(default_factory=RFDETRModelConfig)
    train: RFDETRTrainConfig = Field(default_factory=RFDETRTrainConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


def _load_yaml(path: Path, schema: type[_ConfigT]) -> _ConfigT:
    """Load and validate a YAML file against a Pydantic schema.

    Args:
        path: Path to the YAML config.
        schema: The Pydantic model class to validate against.

    Returns:
        The validated config instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the YAML does not match the schema.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return schema.model_validate(raw)


def load_config(path: Path) -> YOLOExperimentConfig:
    """Load and validate a YOLO experiment config from a YAML file."""
    return _load_yaml(path, YOLOExperimentConfig)


def load_rfdetr_config(path: Path) -> RFDETRExperimentConfig:
    """Load and validate an RF-DETR experiment config from a YAML file."""
    return _load_yaml(path, RFDETRExperimentConfig)


# Tagged union: the ``family`` field selects which schema validates the YAML,
# so a single entrypoint can load either model's config (BasicSR-style dispatch).
AnyExperimentConfig = Annotated[
    YOLOExperimentConfig | RFDETRExperimentConfig,
    Field(discriminator="family"),
]
_EXPERIMENT_ADAPTER: TypeAdapter[YOLOExperimentConfig | RFDETRExperimentConfig] = TypeAdapter(
    AnyExperimentConfig
)


def load_experiment(path: Path) -> YOLOExperimentConfig | RFDETRExperimentConfig:
    """Load any experiment config, dispatching on its ``family`` field.

    Args:
        path: Path to the YAML config. Must contain a top-level ``family`` key
            (``yolo`` or ``rfdetr``).

    Returns:
        The validated config — a :class:`YOLOExperimentConfig` or
        :class:`RFDETRExperimentConfig` depending on ``family``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If ``family`` is missing/unknown or the YAML
            does not match the selected schema.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return _EXPERIMENT_ADAPTER.validate_python(raw)
