"""PersonViT ReID pipeline — train / evaluate / predict (TransReID recipe, native).

Fine-tunes a released PersonViT ReID checkpoint (Hu et al. 2024,
https://github.com/hustvl/PersonViT, Apache-2.0 — ViT-B/16 masked-image-modelling
pretraining on LUPerson, then TransReID fine-tuning; weights at
https://huggingface.co/lakeAGI/PersonViTReID) on our identity-labelled crops.

**Why native and not the upstream trainer** (unlike ``reid_clipreid``, which
subprocesses the official two-stage codebase): the deployed network is a plain
timm ``VisionTransformer`` plus a BNNeck and an identity classifier — TransReID's
``vit_pytorch.py`` is a copy of timm's — and TRACE's inference adapter already
rebuilds it that way. The fine-tune is a single-stage, well-understood recipe
(ID cross-entropy + batch-hard soft-margin triplet on a P x K sampler, SGD with
cosine LR and a long warm-up), so re-implementing it costs less than carrying a
2023 checkout pinned to ``torch.cuda.amp`` and pre-1.0 timm APIs.

**Deployment contract.** ``model.state_dict()`` is saved verbatim, so the
checkpoint keeps the TransReID layout that TRACE's
``apps/trace/worker/reid/personvit.py`` (``backend: personvit``) loads unchanged:

    base.*         # ViT-B/16 backbone, exactly timm's VisionTransformer keys
    bottleneck.*   # BNNeck (BatchNorm1d); TRACE applies it when neck_feat=after
    classifier.*   # identity head — training-only, TRACE skips it

Two things must therefore stay true, and both are asserted by
``tests/test_reid.py``: the backbone is built with the same timm arguments TRACE
uses (no layer-scale, no ``sie_embed`` — SIE camera embeddings are rejected by
that loader), and the input stays at the checkpoint's training resolution
(256x128 -> 16x8 patches -> 129 pos-embed tokens).

**Read-out.** Evaluation returns the POST-BNNeck feature by default
(``model.neck_feat: after``), not the pre-BN class token the upstream test
configs use. The A/B in TRACE's ``docs/report/2026-08-27-personvit-reid-ab.md``
found the pre-BN token has a compressed cosine scale (different-identity mean
~0.73 vs ~0.28 for CLIP-ReID) that puts the production similarity gates inside
its impostor mass; ``after`` is what TRACE deploys, so it is what we score.

Preprocessing mirrors that adapter and the upstream ``INPUT`` block: RGB,
resize to 256x128, normalize mean=std=0.5 — NOT ImageNet statistics. Note the
train transform resizes bicubic and the eval transform bilinear; that asymmetry
is upstream's (``interpolation=3`` only on the train resize) and bilinear is
what TRACE's ``cv2.INTER_LINEAR`` inference path does, so eval matches
deployment rather than training.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import margin_ranking_loss, soft_margin_loss
from torch.utils.data import DataLoader
from torchvision import transforms

from basicdet.metrics.reid import evaluate_retrieval
from basicdet.models.reid_data import (
    CropDataset,
    MarketDataset,
    RandomIdentitySampler,
    ReIDSplit,
    extract_features,
    load_market_dataset,
)
from basicdet.utils import tracking
from basicdet.utils.config import PersonViTExperimentConfig
from basicdet.utils.runtime import resolve_torch_device
from basicdet.utils.seed import set_seed

logger = logging.getLogger(__name__)

RUNS_DIR = Path("runs/reid")

# TransReID INPUT.PIXEL_MEAN / PIXEL_STD for these checkpoints (not ImageNet) —
# identical to the constants in TRACE's personvit adapter.
PIXEL_MEAN = (0.5, 0.5, 0.5)
PIXEL_STD = (0.5, 0.5, 0.5)

# Upstream INPUT.PADDING / INPUT.PROB and SOLVER.MOMENTUM. Fixed rather than
# exposed: they are part of the recipe, not knobs anyone tunes per experiment.
_TRAIN_PADDING = 10
_FLIP_PROB = 0.5
_SGD_MOMENTUM = 0.9

# Keys in a released checkpoint that must NOT be carried into a new fine-tune:
# the identity head is sized to the source dataset's id count, and base.fc is
# TransReID's unused ImageNet head. Same set TRACE's loader drops.
_SKIP_ON_LOAD_PREFIXES = ("classifier.", "base.fc.", "base.head.")

# timm's per-head dim for every ViT size in this family (B 768/12, S 384/6).
_HEAD_DIM = 64


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class PersonViTReID(nn.Module):
    """ViT-B/16 + BNNeck + identity classifier — the TransReID ``build_transformer``.

    Attribute names (``base`` / ``bottleneck`` / ``classifier``) ARE the
    checkpoint schema TRACE loads; renaming any of them breaks deployment.

    Args:
        num_classes: Training identity count (size of the discarded-at-inference
            classifier).
        input_size: ``(H, W)`` crop resolution — must be the checkpoint's
            training resolution, since the position embedding is not resized.
        embed_dim: Backbone width (768 for ViT-B).
        depth: Number of transformer blocks.
        patch_size: Patch side in pixels.
        drop_path_rate: Stochastic-depth rate (upstream MODEL.DROP_PATH).
        neck_feat: Which feature ``forward`` returns in eval mode — ``after``
            (BNNeck output, what TRACE deploys) or ``before`` (pre-BN class
            token, the upstream ``TEST.NECK_FEAT``).

    Raises:
        ValueError: If ``neck_feat`` is not ``before`` or ``after``.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: tuple[int, int] = (256, 128),
        embed_dim: int = 768,
        depth: int = 12,
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
        neck_feat: str = "after",
    ) -> None:
        super().__init__()
        # Lazy import: keeps `--help` and non-ReID families off the timm import.
        from timm.models.vision_transformer import VisionTransformer

        if neck_feat not in ("before", "after"):
            raise ValueError(f"neck_feat must be 'before' or 'after', got {neck_feat!r}")
        self.neck_feat = neck_feat
        self.embed_dim = embed_dim

        # Exactly the arguments TRACE's adapter rebuilds the backbone with, so
        # our state_dict keys are its state_dict keys. Anything that adds
        # parameters (init_values -> layer scale, reg_tokens, SIE) would make
        # the checkpoint unloadable there.
        self.base = VisionTransformer(
            img_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=embed_dim // _HEAD_DIM,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_classes=0,
            class_token=True,
            global_pool="token",
            drop_path_rate=drop_path_rate,
        )
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        # BNNeck: the shift is fixed at 0 so the classifier sees a centred
        # feature (bag-of-tricks); only the scale is learned.
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Embed a batch of crops.

        Args:
            x: ``[batch_size, 3, H, W]`` normalised crops.

        Returns:
            Training mode: ``(logits [B, num_classes], pre-BN feature [B, dim])``
            — the ID loss consumes the logits, the triplet loss the pre-BN
            feature (upstream pairs them exactly this way). Eval mode: the
            ``[B, dim]`` feature selected by ``neck_feat``.
        """
        global_feat = self.base(x)  # forward_features -> norm -> class token
        feat = self.bottleneck(global_feat)
        if self.training:
            return self.classifier(feat), global_feat
        return feat if self.neck_feat == "after" else global_feat


def _infer_architecture(backbone_state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Read ``(embed_dim, depth, patch_size)`` off a backbone state-dict.

    Mirrors TRACE's adapter: the checkpoint, not the config, is the source of
    truth for the architecture, so a mismatch is impossible by construction.

    Args:
        backbone_state: The ``base.``-stripped backbone tensors.

    Returns:
        Tuple of (embedding width, number of blocks, patch side in pixels).

    Raises:
        KeyError: If the state-dict is missing the keys those are read from.
    """
    embed_dim = int(backbone_state["cls_token"].shape[-1])
    depth = 1 + max(int(k.split(".")[1]) for k in backbone_state if k.startswith("blocks."))
    patch_size = int(backbone_state["patch_embed.proj.weight"].shape[-1])
    return embed_dim, depth, patch_size


def _split_checkpoint(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a TransReID-layout checkpoint into backbone and BNNeck tensors.

    Args:
        state: Raw checkpoint mapping.

    Returns:
        ``(backbone_state, bnneck_state)`` with the ``base.``/``bottleneck.``
        prefixes stripped; training-only keys are dropped.

    Raises:
        KeyError: If a key belongs to none of the expected groups.
        ValueError: If the checkpoint carries SIE camera/view embeddings, which
            TRACE's loader refuses.
    """
    backbone: dict[str, torch.Tensor] = {}
    bnneck: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(_SKIP_ON_LOAD_PREFIXES):
            continue
        if key.startswith("bottleneck."):
            bnneck[key.removeprefix("bottleneck.")] = value
        elif key.startswith("base."):
            backbone[key.removeprefix("base.")] = value
        else:
            raise KeyError(f"unexpected checkpoint key {key!r} — not a TransReID layout")
    if any(k.startswith("sie_embed") for k in backbone):
        raise ValueError(
            "checkpoint uses SIE camera/view embeddings — TRACE's personvit loader "
            "rejects those, so such a checkpoint cannot be deployed"
        )
    return backbone, bnneck


def build_model(
    weights: Path,
    num_classes: int,
    input_size: tuple[int, int],
    drop_path_rate: float = 0.1,
    neck_feat: str = "after",
) -> PersonViTReID:
    """Build a :class:`PersonViTReID` warm-started from a checkpoint.

    The backbone and BNNeck are loaded; the identity classifier is left freshly
    initialised at ``num_classes`` (the source checkpoint's head belongs to its
    own identity set). This is how both a fine-tune from a released PersonViT
    ReID checkpoint and a reload of one of our own checkpoints are built.

    Args:
        weights: Path to a TransReID-layout ``.pth``.
        num_classes: Identity count for the classifier head.
        input_size: ``(H, W)`` crop resolution.
        drop_path_rate: Stochastic-depth rate (0 for inference).
        neck_feat: ``after`` (BNNeck output) or ``before`` (pre-BN token).

    Returns:
        The model on CPU, in whatever mode ``nn.Module`` starts in (train).

    Raises:
        FileNotFoundError: If ``weights`` does not exist.
        ValueError: If the checkpoint's position embedding does not match
            ``input_size``, or it carries SIE embeddings.
        RuntimeError: If the backbone or BNNeck state-dict does not match the
            rebuilt modules exactly.
    """
    weights = Path(weights).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(
            f"PersonViT checkpoint not found: {weights}. Released ReID weights: "
            "https://huggingface.co/lakeAGI/PersonViTReID"
        )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    backbone_state, bnneck_state = _split_checkpoint(state)
    embed_dim, depth, patch_size = _infer_architecture(backbone_state)

    tokens = int(backbone_state["pos_embed"].shape[1])
    expected = 1 + (input_size[0] // patch_size) * (input_size[1] // patch_size)
    if tokens != expected:
        raise ValueError(
            f"{weights.name} has {tokens} position-embedding tokens but input_size "
            f"{input_size} with patch {patch_size} needs {expected} — set "
            "model.input_size to the resolution the checkpoint was trained at "
            "(the embedding is deliberately not interpolated: the deployed "
            "adapter would reject a resized one)"
        )

    model = PersonViTReID(
        num_classes=num_classes,
        input_size=input_size,
        embed_dim=embed_dim,
        depth=depth,
        patch_size=patch_size,
        drop_path_rate=drop_path_rate,
        neck_feat=neck_feat,
    )
    missing, unexpected = model.base.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{weights.name}: backbone mismatch missing={missing} unexpected={unexpected}"
        )
    if bnneck_state:
        model.bottleneck.load_state_dict(bnneck_state)
    else:  # a backbone-only checkpoint: BNNeck starts from the identity mapping
        logger.warning("%s has no bottleneck.* — BNNeck initialised fresh", weights.name)
    logger.info(
        "PersonViT: %s -> ViT dim=%d depth=%d patch=%d, %d ids, neck_feat=%s",
        weights.name,
        embed_dim,
        depth,
        patch_size,
        num_classes,
        neck_feat,
    )
    return model


# --------------------------------------------------------------------------- #
# preprocessing (must mirror TRACE's personvit adapter)
# --------------------------------------------------------------------------- #
def build_train_transform(input_size: tuple[int, int], random_erasing: float) -> Callable:
    """Upstream TransReID train augmentation for these checkpoints.

    Order and parameters follow ``datasets/make_dataloader.py``: bicubic resize,
    horizontal flip, 10-px pad + random crop, mean=std=0.5 normalisation, then
    random erasing. ``value="random"`` reproduces timm's ``mode="pixel"``
    erasing (per-pixel noise), which is what upstream uses.

    Args:
        input_size: ``(H, W)`` crop resolution.
        random_erasing: Erasing probability (0 disables).

    Returns:
        A torchvision transform over PIL images.
    """
    h, w = input_size
    ops: list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=_FLIP_PROB),
        transforms.Pad(_TRAIN_PADDING),
        transforms.RandomCrop((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ]
    if random_erasing > 0:
        ops.append(transforms.RandomErasing(p=random_erasing, value="random"))
    return transforms.Compose(ops)


def build_test_transform(input_size: tuple[int, int]) -> Callable:
    """Eval preprocessing: bilinear resize + mean=std=0.5, as TRACE does at inference."""
    h, w = input_size
    return transforms.Compose(
        [
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# losses and schedule
# --------------------------------------------------------------------------- #
def batch_hard_triplet_loss(
    features: torch.Tensor, labels: torch.Tensor, margin: float | None = None
) -> torch.Tensor:
    """Batch-hard triplet loss over a P x K batch.

    For every anchor, the hardest positive (farthest same-identity crop) and the
    hardest negative (nearest different-identity crop) in the batch (Hermans et
    al., "In Defense of the Triplet Loss for Person Re-Identification", 2017).
    ``margin=None`` selects the soft-margin form ``log(1 + exp(d_ap - d_an))``,
    which is what the released PersonViT checkpoints used
    (``MODEL.NO_MARGIN: True``).

    Distances are computed in float32 even under autocast: the squared-distance
    expansion below loses most of its significant digits in fp16 once features
    are ~30 in norm, which is where a ViT class token sits.

    Args:
        features: ``[B, dim]`` pre-BNNeck features.
        labels: ``[B]`` identity labels.
        margin: Hinge margin, or ``None`` for the soft-margin form.

    Returns:
        Scalar loss.

    Raises:
        ValueError: If some anchor has no positive or no negative in the batch
            (a P x K sampler is required — see ``RandomIdentitySampler``).
    """
    features = features.float()
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    if not (same.sum(1) > 1).all() or not (~same).any(1).all():
        raise ValueError(
            "batch-hard mining needs >=2 crops per identity and >=2 identities per "
            "batch — use RandomIdentitySampler with num_instances >= 2"
        )
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, clamped before sqrt as upstream does.
    sq = features.pow(2).sum(1, keepdim=True)
    dist = (sq + sq.t() - 2.0 * features @ features.t()).clamp(min=1e-12).sqrt()

    dist_ap = dist.masked_fill(~same, float("-inf")).max(dim=1).values
    dist_an = dist.masked_fill(same, float("inf")).min(dim=1).values
    target = torch.ones_like(dist_an)
    if margin is None:
        return soft_margin_loss(dist_an - dist_ap, target)
    return margin_ranking_loss(dist_an, dist_ap, target, margin=margin)


def lr_at_epoch(epoch: int, base_lr: float, epochs: int, warmup_epochs: int) -> float:
    """Learning rate for a 0-based epoch index: linear warm-up, then cosine decay.

    Reproduces the timm ``CosineLRScheduler`` configuration upstream uses
    (``solver/scheduler_factory.py``: ``t_initial=epochs``, ``cycle_limit=1``,
    ``warmup_prefix=False``, ``lr_min = 0.002 * base_lr``,
    ``warmup_lr_init = 0.01 * base_lr``) as a pure function of the epoch — no
    scheduler object to keep in step with the loop, and directly testable.

    Note the cosine runs over the full ``[0, epochs)`` span and the warm-up
    *overwrites* its first ``warmup_epochs`` values (``warmup_prefix=False``),
    so the LR jumps slightly at the end of warm-up — upstream behaviour, kept.

    Args:
        epoch: 0-based epoch index.
        base_lr: Peak learning rate.
        epochs: Total epochs (the cosine period).
        warmup_epochs: Linear warm-up length in epochs.

    Returns:
        The learning rate for that epoch.
    """
    lr_min = 0.002 * base_lr
    warmup_lr_init = 0.01 * base_lr
    if epoch < warmup_epochs:
        return warmup_lr_init + epoch * (base_lr - warmup_lr_init) / warmup_epochs
    if epoch >= epochs:
        return lr_min
    return lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * epoch / epochs))


def _build_optimizer(
    model: PersonViTReID,
    base_lr: float,
    bias_lr_factor: float,
    weight_decay: float,
    weight_decay_bias: float,
) -> torch.optim.Optimizer:
    """SGD with upstream's two param groups: biases get a scaled LR and their own decay."""
    biases: list[torch.nn.Parameter] = []
    others: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:  # the BNNeck shift is frozen
            continue
        (biases if name.endswith("bias") else others).append(param)
    # Group order matters: train() rescales group LRs positionally each epoch.
    return torch.optim.SGD(
        [
            {"params": others, "lr": base_lr, "weight_decay": weight_decay},
            {
                "params": biases,
                "lr": base_lr * bias_lr_factor,
                "weight_decay": weight_decay_bias,
            },
        ],
        momentum=_SGD_MOMENTUM,
    )


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def _embed(model: PersonViTReID, split: ReIDSplit, transform: Callable, device: str) -> np.ndarray:
    """Embed one split the way the deployment does.

    ``flip_tta=False``: TRACE's adapter embeds each crop once, so scoring with
    flip augmentation would report a number the deployment cannot reach.
    """
    return extract_features(model, split, transform, device, flip_tta=False)


def _score_retrieval(
    model: PersonViTReID, data: MarketDataset, input_size: tuple[int, int], device: str
) -> dict[str, float]:
    """Embed query + gallery with the deployed read-out and score retrieval."""
    transform = build_test_transform(input_size)
    was_training = model.training
    model.eval()
    query = _embed(model, data.query, transform, device)
    gallery = _embed(model, data.gallery, transform, device)
    if was_training:
        model.train()
    return evaluate_retrieval(
        query,
        data.query.pids,
        data.query.camids,
        gallery,
        data.gallery.pids,
        data.gallery.camids,
    )


def train(config: PersonViTExperimentConfig) -> Path:
    """Fine-tune PersonViT on the configured Market-1501-layout dataset.

    Args:
        config: The validated experiment configuration.

    Returns:
        Path to the final checkpoint
        (``runs/reid/<name>/transformer_last.pth``) — loadable as-is by TRACE's
        ``backend: personvit``.

    Raises:
        FileNotFoundError: If the dataset directory or the starting checkpoint
            is missing.
        ValueError: If the checkpoint does not match ``model.input_size``, or the
            batch/identity settings cannot form a P x K batch.
    """
    set_seed(config.train.seed, deterministic=False)
    device = resolve_torch_device(config.train.device)
    data = load_market_dataset(config.data.dataset_dir)
    out_dir = RUNS_DIR / config.train.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        config.model.weights,
        num_classes=data.num_train_pids,
        input_size=config.model.input_size,
        drop_path_rate=config.model.drop_path,
        neck_feat=config.model.neck_feat,
    ).to(device)

    loader = DataLoader(
        CropDataset(
            data.train, build_train_transform(config.model.input_size, config.train.random_erasing)
        ),
        batch_size=config.train.batch,
        sampler=RandomIdentitySampler(data.train, config.train.batch, config.train.num_instances),
        num_workers=config.train.workers,
        pin_memory=True,
        drop_last=True,  # BatchNorm1d in the BNNeck breaks on a trailing batch of 1
    )
    optimizer = _build_optimizer(
        model,
        config.train.lr,
        config.train.bias_lr_factor,
        config.train.weight_decay,
        config.train.weight_decay_bias,
    )
    amp_enabled = config.train.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)

    use_wandb = tracking.resolve_wandb_enabled(config.wandb)
    if use_wandb:
        import wandb

        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.train.name,
            config=config.model_dump(mode="json"),
        )

    logger.info(
        "PersonViT train: %d ids, %d crops, %d epochs, batch %dx%d, device=%s -> %s",
        data.num_train_pids,
        len(data.train.paths),
        config.train.epochs,
        config.train.batch // config.train.num_instances,
        config.train.num_instances,
        device,
        out_dir,
    )
    for epoch in range(1, config.train.epochs + 1):
        lr = lr_at_epoch(
            epoch - 1, config.train.lr, config.train.epochs, config.train.warmup_epochs
        )
        for group, factor in zip(
            optimizer.param_groups, (1.0, config.train.bias_lr_factor), strict=True
        ):
            group["lr"] = lr * factor

        model.train()
        start = time.time()
        totals = {"loss": 0.0, "id_loss": 0.0, "tri_loss": 0.0, "acc": 0.0}
        n_batches = 0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, features = model(images)
                id_loss = criterion(logits, labels)
                tri_loss = batch_hard_triplet_loss(features, labels, config.train.triplet_margin)
                loss = (
                    config.train.id_loss_weight * id_loss
                    + config.train.triplet_loss_weight * tri_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += float(loss.item())
            totals["id_loss"] += float(id_loss.item())
            totals["tri_loss"] += float(tri_loss.item())
            totals["acc"] += float((logits.argmax(1) == labels).float().mean().item())
            n_batches += 1

        stats = {k: v / max(1, n_batches) for k, v in totals.items()}
        stats.update(epoch=epoch, lr=lr, sec=round(time.time() - start, 1))
        logger.info(
            "epoch %(epoch)d: loss=%(loss).4f id=%(id_loss).4f tri=%(tri_loss).4f "
            "acc=%(acc).4f lr=%(lr).6f %(sec)ss",
            stats,
        )
        if config.train.eval_period and epoch % config.train.eval_period == 0:
            stats.update(_score_retrieval(model, data, config.model.input_size, device))
        if use_wandb:
            wandb.log(stats)
        if epoch % config.train.checkpoint_period == 0:
            torch.save(model.state_dict(), out_dir / f"transformer_{epoch:03d}.pth")

    last = out_dir / "transformer_last.pth"
    torch.save(model.state_dict(), last)
    if use_wandb:
        wandb.finish()
    logger.info("saved %s (TRACE-ready: backend: personvit, weights_path: <this>)", last)
    return last


def evaluate(
    config: PersonViTExperimentConfig,
    weights: str,
    split: str = "test",
    conf: float | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint: mAP / CMC on the query-gallery split.

    Scores the feature ``model.neck_feat`` selects — ``after`` by default, i.e.
    the one TRACE deploys.

    Args:
        config: The experiment configuration (dataset + read-out).
        weights: Path to a ``transformer_*.pth`` checkpoint.
        split: Accepted for registry-signature compatibility; ReID always
            evaluates the query/gallery retrieval split.
        conf: Unused for ReID (registry-signature compatibility).

    Returns:
        ``{"mAP": ..., "rank1": ..., ...}``.
    """
    del split, conf
    device = resolve_torch_device(config.train.device)
    data = load_market_dataset(config.data.dataset_dir)
    # num_classes only sizes the head we are about to ignore; the checkpoint's
    # own head is skipped on load, so any positive value works.
    model = build_model(
        Path(weights),
        num_classes=data.num_train_pids,
        input_size=config.model.input_size,
        drop_path_rate=0.0,
        neck_feat=config.model.neck_feat,
    ).to(device)
    return _score_retrieval(model, data, config.model.input_size, device)


def predict(
    config: PersonViTExperimentConfig,
    weights: str,
    source: Path,
    output: Path,
    conf: float = 0.25,
) -> Any:
    """Embed every crop under ``source``; save features + an index CSV.

    Args:
        config: The experiment configuration (read-out + device).
        weights: Path to a ``transformer_*.pth`` checkpoint.
        source: Directory of person crops.
        output: Output directory for ``features.npy`` + ``index.csv``.
        conf: Unused for ReID (registry-signature compatibility).

    Returns:
        The ``[N, dim]`` feature matrix.
    """
    del conf
    device = resolve_torch_device(config.train.device)
    paths = sorted(Path(source).glob("*.jpg")) + sorted(Path(source).glob("*.png"))
    split = ReIDSplit(paths, np.zeros(len(paths), np.int64), np.zeros(len(paths), np.int64))
    model = build_model(
        Path(weights),
        num_classes=1,
        input_size=config.model.input_size,
        drop_path_rate=0.0,
        neck_feat=config.model.neck_feat,
    ).to(device)
    model.eval()
    feats = _embed(model, split, build_test_transform(config.model.input_size), device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "features.npy", feats)
    (output / "index.csv").write_text("\n".join(str(p) for p in paths) + "\n")
    logger.info("embedded %d crops -> %s", len(paths), output)
    return feats
