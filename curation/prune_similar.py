"""Prune near-duplicate fixed-camera frames to build a deduplicated version.

Fixed-camera video sampled at ~1fps leaves heavy temporal redundancy — a person
barely moving produces near-identical consecutive frames. This tool removes those
near-duplicates from the **MTMDC** portion of a curated version using per-camera
**perceptual-hash (dHash)** greedy dedup: within each camera, frames are walked in
time order and a frame is kept only if it differs from the last kept frame by more
than ``--threshold`` bits (Hamming). Non-MTMDC sources (CrowdHuman / MOT20 — already
diverse and deduped) are kept verbatim.

CLIP/semantic embeddings are deliberately NOT used: on fixed-camera footage they
collapse whole scenes (every frame is semantically ~identical), discarding exactly
the person-motion variation a detector needs. dHash is pixel-level, so it separates
"people moved / appeared" (keep) from "nothing changed" (prune). See curation.md.

Usage (from repo root):
    python -m curation.prune_similar --source v5.1 --out v5.2 --sweep
    python -m curation.prune_similar --source v5.1 --out v5.2 --threshold 6 --build
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from curation import mtmdc

logger = logging.getLogger("mtmdc.prune")

# dHash grid: resize to (HASH_SIZE+1, HASH_SIZE) grayscale, horizontal gradient -> 64 bits.
HASH_SIZE = 8
# Only frames from this source are deduplicated (fixed-camera video). Others are kept as-is.
PRUNE_PREFIX = "mtmdc_"
_CAM_RE = re.compile(r"^(mtmdc_s\d+_c\d+)_")
_FRAME_RE = re.compile(r"_(\d+)\.[a-zA-Z]+$")


def dhash(path: Path) -> np.ndarray:
    """Return a 64-bit dHash as a boolean array. Uses JPEG draft mode for speed."""
    img = Image.open(path)
    img.draft("L", (HASH_SIZE * 4, HASH_SIZE * 4))  # fast approximate downscale during decode
    a = np.asarray(img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _camera_of(name: str) -> str | None:
    m = _CAM_RE.match(name)
    return m.group(1) if m else None


def _frame_of(name: str) -> int:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else 0


def _cache_path(out_dir: Path, split: str) -> Path:
    return out_dir / f"_dhash_{split}.npz"


def compute_hashes(src_dir: Path, out_dir: Path, split: str) -> dict[str, np.ndarray]:
    """dHash every prunable (MTMDC) image in a split; cache to disk for reuse."""
    cache = _cache_path(out_dir, split)
    imgs = sorted(
        p.name for p in (src_dir / "images" / split).iterdir() if p.name.startswith(PRUNE_PREFIX)
    )
    if cache.is_file():
        data = np.load(cache, allow_pickle=True)
        if set(data["names"]) == set(imgs):
            logger.info("[%s] loaded %d cached dHashes", split, len(imgs))
            return dict(zip(data["names"], data["hashes"], strict=True))
    logger.info("[%s] hashing %d MTMDC frames ...", split, len(imgs))
    hashes = {}
    for i, name in enumerate(imgs):
        hashes[name] = dhash(src_dir / "images" / split / name)
        if (i + 1) % 10000 == 0:
            logger.info("  hashed %d/%d", i + 1, len(imgs))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(list(hashes)), hashes=np.array(list(hashes.values())))
    return hashes


def select_kept(src_dir: Path, out_dir: Path, split: str, threshold: int) -> tuple[list[str], int]:
    """Return (kept image names for this split, number of MTMDC frames pruned)."""
    all_imgs = sorted(p.name for p in (src_dir / "images" / split).iterdir())
    non_mtmdc = [n for n in all_imgs if not n.startswith(PRUNE_PREFIX)]
    hashes = compute_hashes(src_dir, out_dir, split)

    by_cam: dict[str, list[str]] = defaultdict(list)
    for name in hashes:
        by_cam[_camera_of(name) or "?"].append(name)

    kept_mtmdc: list[str] = []
    for names in by_cam.values():
        names.sort(key=_frame_of)
        last = None
        for name in names:
            h = hashes[name]
            if last is None or int(np.count_nonzero(h != last)) > threshold:
                kept_mtmdc.append(name)
                last = h
    pruned = len(hashes) - len(kept_mtmdc)
    return non_mtmdc + kept_mtmdc, pruned


def sweep(src_dir: Path, out_dir: Path, thresholds: list[int]) -> None:
    """Report per-split MTMDC prune % across candidate thresholds (no build)."""
    for split in mtmdc.SPLITS:
        n_mtmdc = sum(
            1 for p in (src_dir / "images" / split).iterdir() if p.name.startswith(PRUNE_PREFIX)
        )
        parts = []
        for thr in thresholds:
            _, pruned = select_kept(src_dir, out_dir, split, thr)
            parts.append(f"thr>{thr}: prune {100 * pruned / max(1, n_mtmdc):4.1f}% ({pruned})")
        logger.info("[%s] mtmdc=%d | %s", split, n_mtmdc, "  ".join(parts))


def _filter_coco(src_json: Path, dst_json: Path, kept: set[str]) -> None:
    """Write a COCO json containing only kept images, with contiguous ids."""
    data = json.loads(src_json.read_text())
    new_images: list[dict] = []
    new_anns: list[dict] = []
    old2new: dict[int, int] = {}
    for im in data["images"]:
        if Path(im["file_name"]).name in kept:
            nid = len(new_images) + 1
            old2new[im["id"]] = nid
            new_images.append({**im, "id": nid})
    ann_id = 1
    for a in data["annotations"]:
        if a["image_id"] in old2new:
            new_anns.append({**a, "id": ann_id, "image_id": old2new[a["image_id"]]})
            ann_id += 1
    dst_json.write_text(json.dumps({**data, "images": new_images, "annotations": new_anns}))


def build(source: str, out: str, threshold: int) -> None:
    """Build the pruned version: hardlink kept images+labels, filter COCO + RF-DETR."""
    src_dir = mtmdc.DATA_ROOT / f"persondet_{source}"
    out_dir = mtmdc.DATA_ROOT / f"persondet_{out}"
    stats: dict[str, dict] = {}

    for split in mtmdc.SPLITS:
        kept, pruned = select_kept(src_dir, out_dir, split, threshold)
        kept_set = set(kept)
        rf = mtmdc.RFDETR_SPLIT_DIR[split]
        for sub in (f"images/{split}", f"labels/{split}", f"rfdetr/{rf}"):
            tgt = out_dir / sub
            if tgt.exists():
                shutil.rmtree(tgt)
            tgt.mkdir(parents=True, exist_ok=True)

        for name in kept:
            stem = Path(name).stem
            (out_dir / "images" / split / name).hardlink_to(src_dir / "images" / split / name)
            (out_dir / "rfdetr" / rf / name).hardlink_to(src_dir / "images" / split / name)
            lbl = src_dir / "labels" / split / f"{stem}.txt"
            if lbl.is_file():
                (out_dir / "labels" / split / f"{stem}.txt").hardlink_to(lbl)

        (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
        _filter_coco(
            src_dir / "annotations" / f"instances_{split}.json",
            out_dir / "annotations" / f"instances_{split}.json",
            kept_set,
        )
        _filter_coco(
            src_dir / "rfdetr" / rf / "_annotations.coco.json",
            out_dir / "rfdetr" / rf / "_annotations.coco.json",
            kept_set,
        )
        stats[split] = {"kept": len(kept), "pruned_mtmdc": pruned}
        logger.info("[%s] kept=%d (pruned %d mtmdc dups)", split, len(kept), pruned)

    _write_meta(src_dir, out_dir, source, out, threshold, stats)
    logger.info("Built persondet_%s at %s", out, out_dir)


def _write_meta(
    src_dir: Path, out_dir: Path, source: str, out: str, threshold: int, stats: dict
) -> None:
    yaml_text = (src_dir / "data.yaml").read_text()
    yaml_text = yaml_text.replace(f"persondet_{source}", f"persondet_{out}")
    (out_dir / "data.yaml").write_text(yaml_text)
    lines = [
        f"# persondet_{out}",
        "",
        f"Near-duplicate-pruned copy of **persondet_{source}**: MTMDC fixed-camera frames",
        f"deduplicated per-camera by dHash (Hamming threshold > {threshold}); CrowdHuman/MOT20",
        "kept verbatim. Built by `curation.prune_similar`. Images are hardlinked (no extra disk).",
        "",
        "| split | kept images | mtmdc dups pruned |",
        "|-------|------------:|------------------:|",
    ]
    for split in mtmdc.SPLITS:
        lines.append(f"| {split} | {stats[split]['kept']:,} | {stats[split]['pruned_mtmdc']:,} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", default="v5.1", help="Source version to prune.")
    p.add_argument("--out", default="v5.2", help="Output (pruned) version name.")
    p.add_argument(
        "--threshold", type=int, default=6, help="dHash Hamming distance to treat as a real change."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--sweep", action="store_true", help="Report prune % across thresholds; no build."
    )
    g.add_argument("--build", action="store_true", help="Build the pruned version.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    src_dir = mtmdc.DATA_ROOT / f"persondet_{args.source}"
    out_dir = mtmdc.DATA_ROOT / f"persondet_{args.out}"
    if args.sweep:
        sweep(src_dir, out_dir, thresholds=[2, 4, 6, 8, 10, 14])
    else:
        build(args.source, args.out, args.threshold)


if __name__ == "__main__":
    main()
