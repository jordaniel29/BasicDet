"""ft_net ReID pipeline — train / evaluate / predict (layumi baseline, native).

Trains the ResNet-50 "ft_net" identity-classification baseline (Zheng et al.,
https://github.com/layumi/Person_reID_baseline_pytorch, MIT) and saves
checkpoints whose state-dict keys load UNCHANGED into TRACE's inference
re-implementation (``piapf/reid/ftnet_reid.py``): backbone under ``model.*``,
bottleneck under ``classifier.add_block.{0,1}.*`` (Linear, BatchNorm1d) and the
identity head under ``classifier.classifier.0.*`` (dropped at inference —
TICKET-REID-004, so ``class_num`` never has to match).

Architecture coupling: ``model.stride`` and ``model.linear_num`` in the config
MUST equal the TRACE-side ``ftnet_reid`` config, or the bottleneck keys will
mismatch at deployment (the TRACE loader logs but does not fail).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from basicdet.metrics.reid import evaluate_retrieval
from basicdet.models.reid_data import (
    CropDataset,
    build_test_transform,
    build_train_transform,
    extract_features,
    load_market_dataset,
)
from basicdet.utils import tracking
from basicdet.utils.config import FtNetExperimentConfig
from basicdet.utils.runtime import resolve_torch_device
from basicdet.utils.seed import set_seed

logger = logging.getLogger(__name__)

RUNS_DIR = Path("runs/reid")


class ClassBlock(nn.Module):
    """Bottleneck head: Linear -> BN1d -> Dropout -> classifier Linear.

    Module indices replicate layumi's ``ClassBlock`` so state-dict keys match
    the TRACE loader: ``add_block.0`` (Linear), ``add_block.1`` (BN). Dropout is
    parameter-free, so its presence never affects checkpoint compatibility.
    """

    def __init__(self, input_dim: int, class_num: int, linear: int, droprate: float) -> None:
        super().__init__()
        block: list[nn.Module] = []
        if linear > 0:
            block.append(nn.Linear(input_dim, linear))
        else:
            linear = input_dim
        block.append(nn.BatchNorm1d(linear))
        if droprate > 0:
            block.append(nn.Dropout(p=droprate))
        self.add_block = nn.Sequential(*block)
        self.classifier = nn.Sequential(nn.Linear(linear, class_num))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_block(x)
        return self.classifier(x)


class FtNet(nn.Module):
    """ResNet-50 ft_net (layumi). Attribute names define the checkpoint schema."""

    def __init__(
        self,
        class_num: int,
        stride: int = 2,
        linear_num: int = 512,
        droprate: float = 0.5,
        imagenet_init: bool = True,
    ) -> None:
        super().__init__()
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if imagenet_init else None
        backbone = models.resnet50(weights=weights)
        if stride == 1:  # denser final feature map (bag-of-tricks stride trick)
            backbone.layer4[0].downsample[0].stride = (1, 1)
            backbone.layer4[0].conv2.stride = (1, 1)
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = backbone
        self.classifier = ClassBlock(2048, class_num, linear=linear_num, droprate=droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        x = m.avgpool(x).flatten(1)
        return self.classifier(x)


def _strip_head_for_inference(model: FtNet) -> FtNet:
    """Replace the class head with Identity — forward returns the bottleneck feature."""
    model.classifier.classifier = nn.Identity()
    return model


def _load_checkpoint(config: FtNetExperimentConfig, weights: str, device: str) -> FtNet:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = state.get("model", state) if isinstance(state, dict) else state
    class_num = state["classifier.classifier.0.weight"].shape[0]
    model = FtNet(
        class_num,
        stride=config.model.stride,
        linear_num=config.model.linear_num,
        droprate=0.0,
        imagenet_init=False,
    )
    model.load_state_dict(state, strict=True)
    return _strip_head_for_inference(model).eval().to(device)


def train(config: FtNetExperimentConfig) -> Path:
    """Fine-tune ft_net on the configured Market-1501-layout dataset.

    Args:
        config: The validated experiment configuration.

    Returns:
        Path to the final checkpoint (``runs/reid/<name>/net_last.pth``).

    Raises:
        FileNotFoundError: If the dataset directory is missing.
    """
    set_seed(config.train.seed, deterministic=False)
    device = resolve_torch_device(config.train.device)
    data = load_market_dataset(config.data.dataset_dir)
    out_dir = RUNS_DIR / config.train.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FtNet(
        data.num_train_pids,
        stride=config.model.stride,
        linear_num=config.model.linear_num,
        droprate=config.model.droprate,
        imagenet_init=config.model.imagenet_init,
    ).to(device)

    loader = DataLoader(
        CropDataset(
            data.train,
            build_train_transform(config.model.input_size, config.train.random_erasing),
        ),
        batch_size=config.train.batch,
        shuffle=True,
        num_workers=config.train.workers,
        pin_memory=True,
        drop_last=True,  # BN1d in the bottleneck breaks on a trailing batch of 1
    )

    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": config.train.lr * config.train.backbone_lr_scale},
            {"params": head_params, "lr": config.train.lr},
        ],
        momentum=0.9,
        weight_decay=config.train.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.train.step_lr_epochs, gamma=config.train.step_lr_gamma
    )
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
        "ft_net train: %d ids, %d crops, %d epochs, device=%s -> %s",
        data.num_train_pids,
        len(data.train.paths),
        config.train.epochs,
        device,
        out_dir,
    )
    warmup_iters = config.train.warmup_epochs * max(1, len(loader))
    seen_iters = 0
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        for images, labels in loader:
            if seen_iters < warmup_iters:  # linear LR warm-up (bag-of-tricks)
                scale = (seen_iters + 1) / warmup_iters
                for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = base * scale
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            seen_iters += 1
            running_loss += float(loss.item())
            running_acc += float((logits.argmax(1) == labels).float().mean().item())
            n_batches += 1
        scheduler.step()
        stats = {
            "epoch": epoch,
            "loss": running_loss / max(1, n_batches),
            "acc": running_acc / max(1, n_batches),
            "lr": optimizer.param_groups[-1]["lr"],
            "sec": round(time.time() - t0, 1),
        }
        logger.info("epoch %(epoch)d: loss=%(loss).4f acc=%(acc).4f lr=%(lr).5f %(sec)ss", stats)
        if use_wandb:
            wandb.log(stats)
        if epoch % 10 == 0 or epoch == config.train.epochs:
            torch.save(model.state_dict(), out_dir / f"net_{epoch:03d}.pth")

    last = out_dir / "net_last.pth"
    torch.save(model.state_dict(), last)
    if use_wandb:
        wandb.finish()
    logger.info("saved %s (TRACE-compatible: piapf/reid/ftnet_reid.py)", last)
    return last


def evaluate(
    config: FtNetExperimentConfig,
    weights: str,
    split: str = "test",
    conf: float | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint: mAP / CMC on the query-gallery split.

    Args:
        config: The experiment configuration (dataset + architecture).
        weights: Path to a ``net_*.pth`` checkpoint.
        split: Accepted for registry-signature compatibility; ReID always
            evaluates the query/gallery retrieval split.
        conf: Unused for ReID (registry-signature compatibility).

    Returns:
        ``{"mAP": ..., "rank1": ..., ...}``.
    """
    del split, conf
    device = resolve_torch_device(config.train.device)
    data = load_market_dataset(config.data.dataset_dir)
    model = _load_checkpoint(config, weights, device)
    transform = build_test_transform(config.model.input_size)
    q = extract_features(model, data.query, transform, device)
    g = extract_features(model, data.gallery, transform, device)
    return evaluate_retrieval(
        q, data.query.pids, data.query.camids, g, data.gallery.pids, data.gallery.camids
    )


def predict(
    config: FtNetExperimentConfig,
    weights: str,
    source: Path,
    output: Path,
    conf: float = 0.25,
) -> Any:
    """Embed every crop under ``source``; save features + an index CSV.

    Args:
        config: The experiment configuration (architecture + device).
        weights: Path to a ``net_*.pth`` checkpoint.
        source: Directory of person crops.
        output: Output directory for ``features.npy`` + ``index.csv``.
        conf: Unused for ReID (registry-signature compatibility).

    Returns:
        The ``[N, dim]`` feature matrix.
    """
    del conf
    from basicdet.models.reid_data import ReIDSplit

    device = resolve_torch_device(config.train.device)
    paths = sorted(Path(source).glob("*.jpg")) + sorted(Path(source).glob("*.png"))
    split = ReIDSplit(paths, np.zeros(len(paths), np.int64), np.zeros(len(paths), np.int64))
    model = _load_checkpoint(config, weights, device)
    feats = extract_features(model, split, build_test_transform(config.model.input_size), device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "features.npy", feats)
    (output / "index.csv").write_text("\n".join(str(p) for p in paths) + "\n")
    logger.info("embedded %d crops -> %s", len(paths), output)
    return feats
