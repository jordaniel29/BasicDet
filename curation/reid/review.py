"""Score staged crops against the deployed encoder, so a human reviews the right ones.

Identity labels here come from a tracker, not from a person drawing boxes, so two
failure modes survive into the crops: an id switch mid-track, and a crop where the
target is hidden behind scene geometry (a glass door, a wall) that the MOT
visibility column does not model because nobody is occluding them.

Both show up the same way — the crop's embedding sits far from the rest of its own
identity — so one number finds both: cosine similarity to the identity centroid.
Crops below a threshold are pruned at packaging time; the contact sheets let a
human check the identities themselves before that.

The embedder is TRACE's deployed ``piaspace_clip_reid``, not the training code, so
a review doubles as a check that the checkpoint still loads the way TRACE loads it.
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("curation.reid.review")

EMBED_BATCH = 64
# Calibrated on pia_aihub_v2 with the deployed MSMT17 encoder: the same person
# across scenarios averages 0.936 centroid cosine, two different people 0.543
# (max 0.925). A pair at or above this is worth a human look, never an automatic
# merge — 0.958 once separated two genuinely different colleagues.
SAME_PERSON_SIM = 0.93
# Below this a crop disagrees with its own identity. On the AI-Hub review sheets
# every crop under ~0.45 was a body hidden behind the corridor's glass door.
DEFAULT_PRUNE_SIM = 0.45


def clipreid_embedder(weights: Path, packages: tuple[Path, ...], device: str = "cuda:0") -> Any:
    """Instantiate TRACE's deployed CLIP-ReID embedder (PyTorch path, no TensorRT).

    Args:
        weights: CLIP-ReID ``.pth`` checkpoint.
        packages: Directories to put on ``sys.path``. Both TRACE packages are
            needed: ``piaspace_clip_reid`` imports ``piaspace_trt_runtime`` at
            module load even on the PyTorch path.
        device: Torch device string.

    Raises:
        FileNotFoundError: If a package directory or the checkpoint is absent.
    """
    for pkg in packages:
        if not pkg.is_dir():
            raise FileNotFoundError(f"TRACE package not found: {pkg}")
        if str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))
    if not weights.is_file():
        raise FileNotFoundError(f"CLIP-ReID checkpoint not found: {weights}")
    from piaspace_clip_reid import CLIPReIDEmbedder  # type: ignore[import-not-found]

    return CLIPReIDEmbedder(
        {
            "device": device,
            "input_size": [256, 128],
            "stride": 12,
            "weights_path": str(weights),
        }
    )


def embed(embedder: Any, paths: list[Path], batch: int = EMBED_BATCH) -> np.ndarray:
    """L2-normalised embeddings for ``paths``, shape ``[N, D]``, float32."""
    feats: list[np.ndarray] = []
    for i in range(0, len(paths), batch):
        crops = [cv2.imread(str(p)) for p in paths[i : i + batch]]
        feats.append(np.asarray(embedder.embed(crops), dtype=np.float32))
    f = np.concatenate(feats, axis=0)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)


def centroid(features: np.ndarray) -> np.ndarray:
    """L2-normalised mean of ``features``, shape ``[D]``."""
    c = features.mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


def self_similarity(features: np.ndarray, pids: np.ndarray) -> dict[int, np.ndarray]:
    """Cosine of every crop to its own identity's centroid, grouped by identity."""
    return {
        int(pid): features[pids == pid] @ centroid(features[pids == pid])
        for pid in sorted(set(pids.tolist()))
    }


def coherence_lines(
    features: np.ndarray,
    pids: np.ndarray,
    cameras: np.ndarray,
    groups: np.ndarray | None = None,
) -> list[str]:
    """Human-readable per-identity coherence table plus the closest identity pairs.

    Args:
        features: L2-normalised embeddings ``[N, D]``.
        pids: Identity per crop.
        cameras: Camera per crop.
        groups: Optional label per crop (scenario, session) used to report how
            consistent one identity is across groups — the check for whether a
            reused track id really is the same person.

    Returns:
        Report lines, ready to print or write to a file.
    """
    lines = [f"{'pid':>6} {'crops':>6} {'cams':>5} {'coherence':>10} {'p5':>7} {'min':>7}"]
    cents: dict[int, np.ndarray] = {}
    for pid, sims in self_similarity(features, pids).items():
        sel = pids == pid
        cents[pid] = centroid(features[sel])
        lines.append(
            f"{pid:>6} {int(sel.sum()):>6} {len({*cameras[sel].tolist()}):>5} "
            f"{sims.mean():>10.3f} {np.percentile(sims, 5):>7.3f} {sims.min():>7.3f}"
        )

    if groups is not None:
        lines.append("\nsame identity across groups (same person ~0.94, different ~0.54):")
        for pid in sorted(cents):
            gs = sorted({str(g) for g in groups[pids == pid]})
            for i in range(len(gs)):
                for j in range(i + 1, len(gs)):
                    ci = centroid(features[(pids == pid) & (groups == gs[i])])
                    cj = centroid(features[(pids == pid) & (groups == gs[j])])
                    lines.append(f"  pid {pid}: {gs[i]} vs {gs[j]} = {float(ci @ cj):.3f}")

    lines.append(f"\nclosest different-identity pairs (>= {SAME_PERSON_SIM} means: look at them):")
    ids = sorted(cents)
    pairs = sorted(
        ((float(cents[a] @ cents[b]), a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]),
        reverse=True,
    )
    for sim, a, b in pairs[:10]:
        lines.append(
            f"  pid {a} vs {b}: {sim:.3f}" + ("  <-- CHECK" if sim >= SAME_PERSON_SIM else "")
        )
    return lines


def write_crop_scores(path: Path, names: list[str], scores: list[float]) -> None:
    """Write ``file,self_sim`` so packaging can prune without re-embedding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(("file", "self_sim"))
        w.writerows(sorted(zip(names, scores, strict=True)))


def load_crop_scores(path: Path) -> dict[str, float]:
    """Read the file written by :func:`write_crop_scores`.

    Raises:
        FileNotFoundError: If the review stage has not been run.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing — run the review stage, or disable pruning")
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["file"]: float(r["self_sim"]) for r in csv.DictReader(fh)}


def is_pruned(score: float | None, threshold: float) -> bool:
    """Whether a crop is dropped: only a scored crop, only below a positive threshold.

    An unscored crop is never pruned silently — that would let a partial review
    quietly shrink the dataset.
    """
    return threshold > 0 and score is not None and score < threshold


def contact_sheet(
    rows: list[tuple[Path, int]],
    out_path: Path,
    per_row: int = 24,
    tile: tuple[int, int] = (64, 128),
) -> None:
    """Render one identity as a grid, one row per camera, in frame order.

    Args:
        rows: ``(crop_path, camera)`` for one identity.
        out_path: Destination JPEG.
        per_row: Crops per camera row.
        tile: ``(width, height)`` of each tile.
    """
    by_cam: dict[int, list[Path]] = defaultdict(list)
    for path, cam in rows:
        by_cam[cam].append(path)
    bands: list[np.ndarray] = []
    for cam in sorted(by_cam):
        tiles = [cv2.resize(cv2.imread(str(p)), tile) for p in sorted(by_cam[cam])[:per_row]]
        while len(tiles) < per_row:
            tiles.append(np.full((tile[1], tile[0], 3), 255, np.uint8))
        band = np.hstack(tiles)
        cv2.putText(band, f"cam{cam}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        bands.append(band)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(bands), [cv2.IMWRITE_JPEG_QUALITY, 85])
