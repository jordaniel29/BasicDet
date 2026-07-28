"""Shared ReID data utilities — Market-1501 layout, crop dataset, transforms.

The curated ReID sets (``assets/data/persondet_reid_v*``) use the Market-1501
directory layout so both the native ft_net trainer and the official CLIP-ReID
codebase can consume them without adapters:

    <dataset_dir>/
    ├── bounding_box_train/   # training identities
    ├── query/                # held-out identities, one probe view each
    └── bounding_box_test/    # gallery for the query identities

Filenames encode identity and camera: ``<pid>_c<camid>[s<seq>]_<...>.jpg``
(e.g. ``0042_c15_f0001250.jpg``). ``pid == -1`` marks junk/distractor images
(kept in the gallery per protocol, never used for training).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# `0042_c15_...` / `-1_c3s2_...` — Market-1501 and our curated sets alike.
_NAME_RE = re.compile(r"^(-?\d+)_c(\d+)")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_market_name(path: Path) -> tuple[int, int]:
    """Extract ``(pid, camid)`` from a Market-1501-style filename.

    Args:
        path: Image path whose stem starts with ``<pid>_c<camid>``.

    Returns:
        Tuple of (identity id, camera id). ``pid`` may be ``-1`` (junk).

    Raises:
        ValueError: If the filename does not follow the convention.
    """
    m = _NAME_RE.match(path.stem)
    if m is None:
        raise ValueError(f"not a Market-1501-style name: {path.name}")
    return int(m.group(1)), int(m.group(2))


@dataclass(frozen=True)
class ReIDSplit:
    """One split of identity-labelled crops.

    Attributes:
        paths: Image paths.
        pids: Identity labels aligned with ``paths``.
        camids: Camera ids aligned with ``paths``.
    """

    paths: list[Path]
    pids: np.ndarray
    camids: np.ndarray


@dataclass(frozen=True)
class MarketDataset:
    """A parsed Market-1501-layout dataset.

    Attributes:
        train: Training split (junk ``pid == -1`` removed, pids relabelled to
            contiguous ``0..N-1`` for classification training).
        query: Query split (original pids).
        gallery: Gallery split (original pids; junk entries retained).
        num_train_pids: Number of distinct training identities.
    """

    train: ReIDSplit
    query: ReIDSplit
    gallery: ReIDSplit
    num_train_pids: int


def _scan_split(split_dir: Path) -> ReIDSplit:
    paths = sorted(p for p in split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
    pids, camids = [], []
    for p in paths:
        pid, camid = parse_market_name(p)
        pids.append(pid)
        camids.append(camid)
    return ReIDSplit(paths, np.asarray(pids, dtype=np.int64), np.asarray(camids, dtype=np.int64))


def load_market_dataset(dataset_dir: Path) -> MarketDataset:
    """Scan a Market-1501-layout directory into typed splits.

    Args:
        dataset_dir: Directory with ``bounding_box_train``, ``query`` and
            ``bounding_box_test`` subfolders.

    Returns:
        The parsed dataset; training pids relabelled to contiguous indices.

    Raises:
        FileNotFoundError: If a required split folder is missing.
        ValueError: If the training split is empty.
    """
    for sub in ("bounding_box_train", "query", "bounding_box_test"):
        if not (dataset_dir / sub).is_dir():
            raise FileNotFoundError(f"missing split folder: {dataset_dir / sub}")

    raw_train = _scan_split(dataset_dir / "bounding_box_train")
    keep = raw_train.pids != -1
    kept_paths = [p for p, k in zip(raw_train.paths, keep, strict=True) if k]
    kept_pids = raw_train.pids[keep]
    kept_camids = raw_train.camids[keep]
    if len(kept_paths) == 0:
        raise ValueError(f"no training crops under {dataset_dir / 'bounding_box_train'}")

    unique = np.unique(kept_pids)
    relabel = {int(pid): i for i, pid in enumerate(unique)}
    contiguous = np.asarray([relabel[int(p)] for p in kept_pids], dtype=np.int64)

    train = ReIDSplit(kept_paths, contiguous, kept_camids)
    query = _scan_split(dataset_dir / "query")
    gallery = _scan_split(dataset_dir / "bounding_box_test")
    logger.info(
        "loaded %s: train %d crops / %d ids, query %d, gallery %d",
        dataset_dir.name,
        len(train.paths),
        len(unique),
        len(query.paths),
        len(gallery.paths),
    )
    return MarketDataset(train, query, gallery, num_train_pids=len(unique))


class CropDataset(Dataset):
    """Torch dataset over identity-labelled crops.

    Args:
        split: The crops to serve.
        transform: Torchvision transform applied to each PIL image.
    """

    def __init__(self, split: ReIDSplit, transform: Callable) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.split.paths[idx]) as im:
            img = self.transform(im.convert("RGB"))
        return img, int(self.split.pids[idx])


def build_train_transform(input_size: tuple[int, int], random_erasing: float) -> Callable:
    """Standard ReID training augmentation (layumi/BoT recipe)."""
    h, w = input_size
    ops: list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if random_erasing > 0:
        ops.append(transforms.RandomErasing(p=random_erasing, value=0))
    return transforms.Compose(ops)


def build_test_transform(input_size: tuple[int, int]) -> Callable:
    """Deterministic eval-time preprocessing (must mirror TRACE inference)."""
    h, w = input_size
    return transforms.Compose(
        [
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    split: ReIDSplit,
    input_size: tuple[int, int],
    device: str,
    batch_size: int = 64,
    workers: int = 4,
    flip_tta: bool = True,
) -> np.ndarray:
    """Embed a split with a torch model, mirroring TRACE inference behaviour.

    Adds horizontally-flipped features when ``flip_tta`` (the ft_net runtime
    default) and L2-normalises the result.

    Args:
        model: Feature extractor mapping ``[B, 3, H, W]`` to ``[B, dim]``.
        split: Crops to embed.
        input_size: ``(H, W)`` model input resolution.
        device: Torch device string.
        batch_size: Inference batch size.
        workers: Dataloader workers.
        flip_tta: Add flipped-image features before normalisation.

    Returns:
        ``[len(split), dim]`` float32 L2-normalised features.
    """
    loader = DataLoader(
        CropDataset(split, build_test_transform(input_size)),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )
    model.eval()
    feats: list[np.ndarray] = []
    for batch, _ in loader:
        batch = batch.to(device)
        out = model(batch)
        if flip_tta:
            out = out + model(torch.flip(batch, dims=[3]))
        out = torch.nn.functional.normalize(out.float(), p=2, dim=1)
        feats.append(out.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)
