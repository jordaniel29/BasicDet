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
        cache: Image caching to avoid re-decoding JPEGs every epoch — ``"ram"``
            (fastest, needs RAM), ``"disk"`` (cached .npy), or ``False`` (off).
            Big speedup when training is data-loading bound.
        deterministic: Force deterministic algorithms (reproducible but slower);
            ``False`` enables cuDNN autotuning for speed.
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
    cache: bool | str = False
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
        variant: Model size — ``nano`` / ``small`` / ``medium`` / ``base`` /
            ``large`` (smaller = faster, less accurate).
        resolution: Square input resolution in pixels; must be divisible by 56.
            ``None`` uses the variant's default.
        num_classes: Number of classes. ``None`` lets RF-DETR infer from the
            dataset (correct for this single-class ``person`` set).
    """

    variant: Literal["nano", "small", "medium", "base", "large"] = "base"
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


# --------------------------------------------------------------------------- #
# ReID (person re-identification embedders for the downstream TRACE tracker)
#
# ReID consumes identity-labelled person crops in the Market-1501 directory
# layout (``bounding_box_train/``, ``query/``, ``bounding_box_test/``; filenames
# ``<pid>_c<camid>_...jpg``), not detection boxes. Two families:
#   - ``reid_ftnet``:    layumi ft_net (ResNet-50 + bottleneck head) trained
#                        natively here; checkpoint drops into TRACE's
#                        ``piapf/reid/ftnet_reid.py`` loader unchanged.
#   - ``reid_clipreid``: orchestrates the official CLIP-ReID two-stage trainer
#                        (third_party checkout); the output ``.pth`` is consumed
#                        verbatim by TRACE's ``piaspace_clip_reid`` package.
# --------------------------------------------------------------------------- #


class ReIDDataConfig(BaseModel):
    """Dataset location for ReID (Market-1501 directory layout).

    Attributes:
        dataset_dir: Directory containing ``bounding_box_train/``, ``query/``
            and ``bounding_box_test/`` folders of identity-labelled crops named
            ``<pid>_c<camid>_<...>.jpg``.
    """

    dataset_dir: Path


class FtNetModelConfig(BaseModel):
    """ft_net architecture settings — MUST match the TRACE inference config.

    Attributes:
        linear_num: Bottleneck feature dim (``0`` = use the raw 2048-d pooled
            feature). TRACE's ``ftnet_reid.py`` default is 512.
        stride: Stride of the last ResNet block (``1`` = denser map, BoT trick).
            TRACE's default is 2.
        droprate: Dropout after the bottleneck BN during training (parameter-free
            at inference; does not affect checkpoint compatibility).
        imagenet_init: Start from ImageNet-pretrained ResNet-50 (standard).
        input_size: Crop input size ``[H, W]`` — must match TRACE inference
            (256x128).
    """

    linear_num: int = 512
    stride: int = 2
    droprate: float = 0.5
    imagenet_init: bool = True
    input_size: tuple[int, int] = (256, 128)


class ReIDTrainConfig(BaseModel):
    """ReID training-loop hyperparameters (ft_net native trainer).

    Attributes:
        epochs: Training epochs (layumi baseline: 60).
        batch: Crops per batch.
        lr: Learning rate for NEW parameters (bottleneck + classifier); the
            pretrained backbone uses ``lr * backbone_lr_scale``.
        backbone_lr_scale: Backbone LR multiplier (layumi baseline: 0.1).
        weight_decay: SGD weight decay.
        label_smoothing: Cross-entropy label smoothing.
        step_lr_epochs: Decay LR by ``step_lr_gamma`` every N epochs.
        step_lr_gamma: LR decay factor.
        warmup_epochs: Linear LR warm-up epochs.
        random_erasing: Random-erasing probability (0 disables).
        workers: Dataloader worker processes.
        seed: Global RNG seed.
        device: ``"auto"``, ``"cpu"``, or a CUDA index string.
        name: Run name; outputs land in ``runs/reid/<name>/``.
    """

    epochs: int = 60
    batch: int = 32
    lr: float = 0.05
    backbone_lr_scale: float = 0.1
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    step_lr_epochs: int = 20
    step_lr_gamma: float = 0.1
    warmup_epochs: int = 5
    random_erasing: float = 0.5
    workers: int = 8
    seed: int = 42
    device: str = "auto"
    name: str = "ftnet_person"


class FtNetExperimentConfig(BaseModel):
    """Full configuration for an ft_net ReID fine-tune.

    Attributes:
        family: Discriminator selecting the ft_net pipeline.
    """

    family: Literal["reid_ftnet"] = "reid_ftnet"
    data: ReIDDataConfig
    model: FtNetModelConfig = Field(default_factory=FtNetModelConfig)
    train: ReIDTrainConfig = Field(default_factory=ReIDTrainConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class ClipReIDModelConfig(BaseModel):
    """CLIP-ReID architecture settings — must match the deployed TRT encoder.

    Attributes:
        backbone: CLIP vision backbone (deployed engine is ViT-B-16).
        stride: Patch-embedding stride (deployed engine: 12 — overlapping
            patches).
        sie_camera: Enable Side Information Embeddings over camera ids during
            training (the deployed ``12x12sie`` checkpoints used this;
            inference ignores SIE).
        input_size: Crop input size ``[H, W]`` (deployed engine: 256x128).
    """

    backbone: Literal["ViT-B-16"] = "ViT-B-16"
    stride: int = 12
    sie_camera: bool = True
    input_size: tuple[int, int] = (256, 128)


class ClipReIDTrainConfig(BaseModel):
    """CLIP-ReID two-stage training settings (forwarded to the official repo).

    Attributes:
        stage1_epochs: Prompt-learning stage epochs (official person cfg: 120
            iterations-based; see repo configs).
        stage2_epochs: Image-encoder fine-tune epochs (official: 60).
        batch: Stage-2 batch size (P*K sampler; official person cfg: 64).
        num_instances: Crops per identity in a batch (K of the PK sampler).
        base_lr_stage2: Stage-2 base learning rate.
        pretrain_weights: Starting checkpoint — path to a CLIP-ReID ``.pth``
            (e.g. the deployed MSMT17 one, to fine-tune from it) or ``null`` to
            start from OpenAI CLIP weights as in the official recipe.
        workers: Dataloader worker processes.
        seed: Global RNG seed.
        device: ``"auto"``, ``"cpu"``, or a CUDA index string.
        name: Run name; outputs land in ``runs/reid/<name>/``.
        piaspace_pkg: Path to TRACE's ``piaspace-clip-reid`` package ``src/``
            dir — evaluation embeds through the DEPLOYED encoder so every eval
            doubles as a deployment-contract test.
        extra: Extra ``KEY: value`` overrides merged verbatim onto the official
            repo's yacs config (dotted keys, e.g. ``SOLVER.STAGE2.IMS_PER_BATCH``).
    """

    stage1_epochs: int = 120
    stage2_epochs: int = 60
    batch: int = 64
    num_instances: int = 4
    base_lr_stage2: float = 5e-6
    pretrain_weights: Path | None = None
    workers: int = 8
    seed: int = 42
    device: str = "auto"
    name: str = "clipreid_person"
    piaspace_pkg: Path = Path(
        "/home/jordan/jordan/TRACE_SSAVE-AI-MVP/packages/piaspace-clip-reid/src"
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class ClipReIDExperimentConfig(BaseModel):
    """Full configuration for a CLIP-ReID fine-tune.

    Attributes:
        family: Discriminator selecting the CLIP-ReID pipeline.
    """

    family: Literal["reid_clipreid"] = "reid_clipreid"
    data: ReIDDataConfig
    model: ClipReIDModelConfig = Field(default_factory=ClipReIDModelConfig)
    train: ClipReIDTrainConfig = Field(default_factory=ClipReIDTrainConfig)
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
AnyExperiment = (
    YOLOExperimentConfig | RFDETRExperimentConfig | FtNetExperimentConfig | ClipReIDExperimentConfig
)
AnyExperimentConfig = Annotated[AnyExperiment, Field(discriminator="family")]
_EXPERIMENT_ADAPTER: TypeAdapter[AnyExperiment] = TypeAdapter(AnyExperimentConfig)


def load_experiment(path: Path) -> AnyExperiment:
    """Load any experiment config, dispatching on its ``family`` field.

    Args:
        path: Path to the YAML config. Must contain a top-level ``family`` key
            (``yolo`` / ``rfdetr`` / ``reid_ftnet`` / ``reid_clipreid``).

    Returns:
        The validated config for the matching family.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If ``family`` is missing/unknown or the YAML
            does not match the selected schema.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return _EXPERIMENT_ADAPTER.validate_python(raw)
