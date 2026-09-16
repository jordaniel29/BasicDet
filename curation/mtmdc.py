"""Shared constants and helpers for curating the MTMDC multi-sensor source.

Source: ``06_multisensor_trajectory_tracking`` (NIA MTMDC) — 22 scenarios, each
with 16 fixed-camera 1920x1080 @30fps ``.avi`` videos plus a per-camera VATIC
``.txt`` (already generated from the per-frame JSON labels by ``json_to_vatic.py``
that ships with the raw data). See ``curation.md`` for the full playbook.

Two camera layouts are reused across scenarios:
    * Space A (indoor logistics facility): scenarios 01, 10-19.
    * Space B (outdoor plaza + buildings): scenarios 31-39, 42-43.

This module centralises every tunable parameter and the format conversions so the
extract / build / verify scripts share one source of truth (CLAUDE.md: no magic
numbers in logic).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# Raw source (read-only — never modified).
RAW_ROOT = Path.home() / "jordan/ai_public/tracking_dataset/06_multisensor_trajectory_tracking"
# Where curated datasets and the shared frame pool live (local SSD, fast train I/O).
# Repo-relative: datasets moved to the per-task ``assets/data/detection/`` layout,
# and the previous ``~/jordan/Person-Det`` absolute path no longer exists. Deriving
# this from the file location keeps every curation script working after a move.
DATA_ROOT = Path(__file__).resolve().parent.parent / "assets/data/detection"
# Shared, deduplicated-by-subsampling frame pool reused by every v4 version.
# Images are hardlinked from here into each version (same filesystem -> no copy).
FRAME_POOL = DATA_ROOT / "_mtmdc_frames"

# --- Provenance / naming -----------------------------------------------------
# Short source tag prefixed onto every filename for global uniqueness and
# traceability (curation.md sec.1). New tag for this source: MTMDC.
SOURCE_TAG = "mtmdc"

# --- Curation parameters -----------------------------------------------------
# All MTMDC videos are 30fps.
SOURCE_FPS = 30
# Pool extraction stride: decode every Nth frame into the shared pool. 30fps/15 ==
# 2fps. Kept dense so the dataset stride can be re-tuned without re-decoding.
SUBSAMPLE_EVERY = 15
# Dataset build stride (original-frame units; must be a multiple of
# SUBSAMPLE_EVERY). Datasets keep every BUILD_STRIDE-th frame -> SOURCE_FPS /
# BUILD_STRIDE effective fps. Coarser than the pool because these fixed cameras
# produce heavy near-duplicate redundancy; 1fps preserves diversity on the
# dynamic multi-person scenes while halving the frame count. For fixed cameras
# this temporal subsampling *is* the deduplication (curation.md sec.4 + sec.5).
BUILD_STRIDE = 30
# Deterministic train/val split within the train/val pool.
SEED = 42
VAL_FRACTION = 0.10
# Boxes with width or height <= this many pixels are dropped (curation.md sec.3).
MIN_BOX_PX = 1.0
# Drop frames with no labeled person instead of keeping them as background
# negatives. MTMDC only annotates the tracked-capture subjects, so ~50% of its
# zero-label frames actually contain real *unlabeled* bystanders (verified with a
# COCO person detector). Keeping them would train the detector to suppress real
# people — exactly what curation.md sec.5's "verify negatives are truly
# people-free" guards against. Labeled frames are exhaustive and reliable, so we
# keep only those. Flip to True only if a future source has clean negatives.
KEEP_NEGATIVES = False
# Single-class output.
PERSON_CLASS_ID = 0  # YOLO class id
PERSON_COCO_CATEGORY = {"id": 1, "name": "person", "supercategory": "person"}
JPEG_QUALITY = 95

# --- Dataset layout ----------------------------------------------------------
SPLITS = ("train", "val", "test")
# RF-DETR uses "valid" for its validation subfolder (see basicdet/models/rfdetr.py).
RFDETR_SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}

# --- Scenario / camera topology ---------------------------------------------
SPACE_A_SCENARIOS: tuple[int, ...] = (1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
SPACE_B_SCENARIOS: tuple[int, ...] = (31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43)
ALL_SCENARIOS: tuple[int, ...] = tuple(sorted(SPACE_A_SCENARIOS + SPACE_B_SCENARIOS))
CAMERAS: tuple[int, ...] = tuple(range(1, 17))  # camera01 .. camera16

# --- Per-version split rules -------------------------------------------------
# v4.1: camera-disjoint *within every scenario* — cams 15-16 are the held-out
#       test cameras; cams 1-14 form the train/val pool.
V41_TEST_CAMERAS: frozenset[int] = frozenset({15, 16})
# v4.2: scenario-disjoint — these whole scenarios are the held-out test set.
V42_TEST_SCENARIOS: frozenset[int] = frozenset({18, 19, 42, 43})

FRAME_RE = re.compile(r"_(\d+)\.json$")

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel corners


# --- Naming helpers ----------------------------------------------------------
def scenario_name(scenario: int) -> str:
    """``1`` -> ``'scenario01'`` (matches the raw folder names)."""
    return f"scenario{scenario:02d}"


def camera_name(camera: int) -> str:
    """``1`` -> ``'camera01'`` (matches the raw folder/file names)."""
    return f"camera{camera:02d}"


def frame_stem(scenario: int, camera: int, frame: int) -> str:
    """Globally-unique filename stem encoding source+scenario+camera+frame.

    e.g. ``mtmdc_s01_c15_000150`` — lets a downstream consumer recover the
    scenario/camera/frame and split by them (curation.md sec.1).
    """
    return f"{SOURCE_TAG}_s{scenario:02d}_c{camera:02d}_{frame:06d}"


STEM_RE = re.compile(rf"^{SOURCE_TAG}_s(\d+)_c(\d+)_(\d+)$")


def parse_stem(stem: str) -> tuple[int, int, int]:
    """``'mtmdc_s01_c15_000150'`` -> ``(scenario, camera, frame)``.

    Raises:
        ValueError: If the stem does not match the naming convention.
    """
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(f"unexpected filename stem: {stem!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def space_of(scenario: int) -> str:
    """Return the camera-layout space (``'A'`` or ``'B'``) for a scenario."""
    if scenario in SPACE_A_SCENARIOS:
        return "A"
    if scenario in SPACE_B_SCENARIOS:
        return "B"
    raise ValueError(f"unknown scenario {scenario}")


# --- Raw-data access ---------------------------------------------------------
def camera_video(scenario: int, camera: int) -> Path:
    """Path to ``scenario<S>/camera<C>.avi``."""
    return RAW_ROOT / scenario_name(scenario) / f"{camera_name(camera)}.avi"


def camera_vatic(scenario: int, camera: int) -> Path:
    """Path to ``scenario<S>/camera<C>.txt`` (VATIC annotations)."""
    return RAW_ROOT / scenario_name(scenario) / f"{camera_name(camera)}.txt"


def camera_json_dir(scenario: int, camera: int) -> Path:
    """Path to ``scenario<S>/camera<C>/`` (per-frame JSON labels)."""
    return RAW_ROOT / scenario_name(scenario) / camera_name(camera)


def annotated_frame_indices(scenario: int, camera: int) -> list[int]:
    """Frame indices that have a per-frame JSON (the annotated universe).

    Includes frames with zero person boxes (kept as background negatives when
    subsampled — curation.md sec.5). The JSON filename index equals the 0-based
    video decode index (verified by round-trip overlay).

    Returns:
        Sorted list of frame indices present under the camera's JSON folder.
    """
    cam_dir = camera_json_dir(scenario, camera)
    indices: list[int] = []
    for p in cam_dir.glob("*.json"):
        m = FRAME_RE.search(p.name)
        if m:
            indices.append(int(m.group(1)))
    indices.sort()
    return indices


def parse_vatic(path: Path) -> dict[int, list[Box]]:
    """Parse a VATIC ``.txt`` into ``frame -> [ (x1,y1,x2,y2), ... ]``.

    VATIC columns: ``track_id xmin ymin xmax ymax frame lost occluded generated
    "LABEL"``. Every row in this source is a (possibly occluded) ``PERSON`` with
    ``lost==0``, so all rows are kept; geometry cleaning happens at box-conversion
    time. Frames absent from the file have no person boxes (negatives).

    Args:
        path: Path to a per-camera VATIC ``.txt``.

    Returns:
        Mapping from frame index to its list of pixel-corner boxes.
    """
    boxes: dict[int, list[Box]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 10:
                continue
            x1, y1, x2, y2 = (float(parts[i]) for i in (1, 2, 3, 4))
            frame = int(float(parts[5]))
            boxes[frame].append((x1, y1, x2, y2))
    return boxes


# --- Box geometry ------------------------------------------------------------
def clamp_box(box: Box, width: int, height: int) -> Box:
    """Clamp pixel corners to the image bounds ``[0,W] x [0,H]``."""
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def is_valid_box(box: Box) -> bool:
    """True if the box has width and height above ``MIN_BOX_PX``."""
    x1, y1, x2, y2 = box
    return (x2 - x1) > MIN_BOX_PX and (y2 - y1) > MIN_BOX_PX


def box_to_yolo(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    """Pixel corners -> normalised YOLO ``(cx, cy, w, h)`` in ``[0,1]``."""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        bw / width,
        bh / height,
    )


def box_to_coco(box: Box) -> tuple[list[int], int]:
    """Pixel corners -> COCO ``(bbox=[x,y,w,h], area)`` in absolute pixels.

    Returns integer pixels to match the existing combined_v* datasets (the source
    VATIC coordinates are already integer-valued).
    """
    x1, y1, x2, y2 = box
    bx, by = round(x1), round(y1)
    bw, bh = round(x2 - x1), round(y2 - y1)
    return [bx, by, bw, bh], bw * bh


def clean_boxes(raw: list[Box], width: int, height: int) -> list[Box]:
    """Clamp to bounds and drop degenerate boxes (curation.md sec.3)."""
    cleaned: list[Box] = []
    for b in raw:
        cb = clamp_box(b, width, height)
        if is_valid_box(cb):
            cleaned.append(cb)
    return cleaned


# --- Filesystem / IO ---------------------------------------------------------
def hardlink(src: Path, dst: Path) -> None:
    """Hardlink ``src`` -> ``dst`` (idempotent; copies across filesystems).

    Hardlinks share bytes, so images can live in many datasets at one disk cost
    (curation.md sec.1). Falls back to a copy when src/dst are on different
    filesystems.
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_json(path: Path, obj: dict) -> None:
    """Write ``obj`` as compact JSON to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
