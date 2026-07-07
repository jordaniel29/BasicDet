"""Stages 3 + 5 — assemble a curated version from the shared frame pool.

Given the extracted frame pool + ``manifest.csv`` (see ``extract_frames.py``),
this assigns each frame to train/val/test for the requested version, hardlinks
the pooled JPEGs into the version's split folders, and writes the YOLO labels,
COCO JSONs, the RF-DETR per-folder layout, ``data.yaml`` and ``README.md``.

Two versions share the same pool and differ only in their split rule:

    v4.1  camera-disjoint within every scenario:
          cameras 15-16 -> test; cameras 1-14 -> train/val pool.
    v4.2  scenario-disjoint:
          scenarios 18,19,42,43 (all cameras) -> test; the rest -> train/val pool.

In both, the train/val pool is split 90/10 by a deterministic seeded shuffle
(curation.md sec.5). The held-out test set is the honest generalization probe
(unseen cameras for v4.1, unseen scenes for v4.2).

The VATIC ``.txt`` is the single source of truth for boxes; YOLO (normalised) and
COCO (absolute px) are both derived from it.

Usage (from the repo root):
    python -m curation.build_version v4.1
    python -m curation.build_version v4.2
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from curation import mtmdc

logger = logging.getLogger("mtmdc.build")

# Re-exported from mtmdc so verify_version (which imports them here) and the merge
# tool share one definition of the dataset layout.
SPLITS = mtmdc.SPLITS
RFDETR_SPLIT_DIR = mtmdc.RFDETR_SPLIT_DIR


@dataclass(frozen=True)
class FrameRecord:
    """One extracted frame, as listed in the pool manifest."""

    scenario: int
    camera: int
    frame: int
    width: int
    height: int
    rel_path: str  # relative to FRAME_POOL

    @property
    def stem(self) -> str:
        return mtmdc.frame_stem(self.scenario, self.camera, self.frame)

    @property
    def pool_jpg(self) -> Path:
        return mtmdc.FRAME_POOL / self.rel_path


def load_manifest() -> list[FrameRecord]:
    """Read the pool manifest into typed records.

    Raises:
        FileNotFoundError: If the manifest is missing (run extract_frames first).
    """
    manifest = mtmdc.FRAME_POOL / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest} not found — run `python -m curation.extract_frames` first."
        )
    records: list[FrameRecord] = []
    with open(manifest, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append(
                FrameRecord(
                    scenario=int(row["scenario"]),
                    camera=int(row["camera"]),
                    frame=int(row["frame"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    rel_path=row["rel_path"],
                )
            )
    return records


def is_test(version: str, rec: FrameRecord) -> bool:
    """Whether a frame belongs to the held-out test set for this version."""
    match version:
        case "v4.1":
            return rec.camera in mtmdc.V41_TEST_CAMERAS
        case "v4.2":
            return rec.scenario in mtmdc.V42_TEST_SCENARIOS
        case _:
            raise ValueError(f"unknown version {version!r} (expected v4.1 or v4.2)")


def select_stride(records: list[FrameRecord], stride: int) -> list[FrameRecord]:
    """Keep every ``stride``-th original frame (a coarser subset of the pool).

    ``stride`` must be a multiple of the pool's ``SUBSAMPLE_EVERY`` so the kept
    frames exist in the pool. ``stride == SUBSAMPLE_EVERY`` keeps everything.

    Raises:
        ValueError: If ``stride`` is not a positive multiple of SUBSAMPLE_EVERY.
    """
    if stride <= 0 or stride % mtmdc.SUBSAMPLE_EVERY != 0:
        raise ValueError(
            f"stride must be a positive multiple of SUBSAMPLE_EVERY="
            f"{mtmdc.SUBSAMPLE_EVERY}, got {stride}"
        )
    return [r for r in records if r.frame % stride == 0]


def assign_splits(version: str, records: list[FrameRecord]) -> dict[str, list[FrameRecord]]:
    """Partition records into train/val/test for the given version.

    Test is the version's held-out cameras/scenarios; the remainder is split
    90/10 train/val by a deterministic seeded shuffle (curation.md sec.5).
    """
    test = [r for r in records if is_test(version, r)]
    pool = [r for r in records if not is_test(version, r)]

    # Sort first for a stable, order-independent shuffle, then seed.
    pool.sort(key=lambda r: r.rel_path)
    rng = random.Random(mtmdc.SEED)
    rng.shuffle(pool)
    n_val = round(len(pool) * mtmdc.VAL_FRACTION)
    val = pool[:n_val]
    train = pool[n_val:]
    return {"train": train, "val": val, "test": test}


def _reset_dirs(version_dir: Path) -> None:
    """Remove previously-built outputs (but never the shared pool)."""
    for sub in ("images", "labels", "annotations", "rfdetr"):
        target = version_dir / sub
        if target.exists():
            shutil.rmtree(target)
    for split in SPLITS:
        (version_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (version_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (version_dir / "rfdetr" / RFDETR_SPLIT_DIR[split]).mkdir(parents=True, exist_ok=True)
    (version_dir / "annotations").mkdir(parents=True, exist_ok=True)


def _write_yolo_label(path: Path, boxes: list[mtmdc.Box], w: int, h: int) -> None:
    """Write a YOLO ``.txt`` (one ``0 cx cy w h`` line per box; empty = negative)."""
    lines = []
    for b in boxes:
        cx, cy, bw, bh = mtmdc.box_to_yolo(b, w, h)
        lines.append(f"{mtmdc.PERSON_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _boxes_for_camera(scenario: int, camera: int) -> dict[int, list[mtmdc.Box]]:
    """Parse one camera's VATIC once -> frame -> raw pixel-corner boxes."""
    return mtmdc.parse_vatic(mtmdc.camera_vatic(scenario, camera))


def build(version: str, stride: int = mtmdc.BUILD_STRIDE) -> dict:
    """Build the version end-to-end and return a stats dict for the README/log."""
    version_dir = mtmdc.DATA_ROOT / f"persondet_{version}"
    pool = load_manifest()
    records = select_stride(pool, stride)
    splits = assign_splits(version, records)
    logger.info(
        "%s: stride=%d (%.2g fps), %d/%d pool frames -> train=%d val=%d test=%d",
        version,
        stride,
        mtmdc.SOURCE_FPS / stride,
        len(records),
        len(pool),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    _reset_dirs(version_dir)

    # Cache VATIC per camera so each .txt is parsed at most once across all splits.
    vatic_cache: dict[tuple[int, int], dict[int, list[mtmdc.Box]]] = {}

    def boxes_of(rec: FrameRecord) -> list[mtmdc.Box]:
        key = (rec.scenario, rec.camera)
        if key not in vatic_cache:
            vatic_cache[key] = _boxes_for_camera(*key)
        raw = vatic_cache[key].get(rec.frame, [])
        return mtmdc.clean_boxes(raw, rec.width, rec.height)

    per_split: dict[str, dict[str, int]] = {}

    for split, recs in splits.items():
        recs_sorted = sorted(recs, key=lambda r: r.stem)
        coco: dict[str, list[dict]] = {
            "images": [],
            "annotations": [],
            "categories": [mtmdc.PERSON_COCO_CATEGORY],
        }
        ann_id = 1
        image_id = 0
        n_boxes = n_neg = 0
        img_dir = version_dir / "images" / split
        lbl_dir = version_dir / "labels" / split
        rf_dir = version_dir / "rfdetr" / RFDETR_SPLIT_DIR[split]

        for rec in recs_sorted:
            boxes = boxes_of(rec)
            # MTMDC zero-label frames are contaminated with unlabeled bystanders;
            # drop them rather than ship them as background negatives (see mtmdc.py).
            if not boxes and not mtmdc.KEEP_NEGATIVES:
                continue
            if not boxes:
                n_neg += 1

            image_id += 1
            fname = f"{rec.stem}.jpg"
            mtmdc.hardlink(rec.pool_jpg, img_dir / fname)
            mtmdc.hardlink(rec.pool_jpg, rf_dir / fname)
            _write_yolo_label(lbl_dir / f"{rec.stem}.txt", boxes, rec.width, rec.height)

            coco["images"].append(
                {"id": image_id, "file_name": fname, "width": rec.width, "height": rec.height}
            )
            for b in boxes:
                bbox, area = mtmdc.box_to_coco(b)
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": mtmdc.PERSON_COCO_CATEGORY["id"],
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                n_boxes += 1

        # COCO is written to both the standard location and the RF-DETR folder.
        mtmdc.write_json(version_dir / "annotations" / f"instances_{split}.json", coco)
        mtmdc.write_json(rf_dir / "_annotations.coco.json", coco)
        per_split[split] = {"images": image_id, "boxes": n_boxes, "negatives": n_neg}
        logger.info("  %-5s: %d images, %d boxes, %d negatives", split, image_id, n_boxes, n_neg)

    stats = {
        "version": version,
        "per_split": per_split,
        "scenarios": sorted({r.scenario for r in records}),
        "cameras": sorted({r.camera for r in records}),
    }

    _write_data_yaml(version_dir, stride)
    _write_readme(version_dir, version, stats, stride)
    logger.info("%s built at %s", version, version_dir)
    return stats


def _write_data_yaml(version_dir: Path, stride: int) -> None:
    """Ultralytics YOLO dataset config (curation.md sec.1)."""
    content = (
        f"# {version_dir.name} — MTMDC person detection (fixed-camera, subsampled "
        f"every {stride}th frame, {mtmdc.SOURCE_FPS / stride:.2g} fps)\n"
        f"path: {version_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names: ['person']\n"
    )
    (version_dir / "data.yaml").write_text(content, encoding="utf-8")


def _write_readme(version_dir: Path, version: str, stats: dict, stride: int) -> None:
    split_desc = {
        "v4.1": (
            "Camera-disjoint **within every scenario**: cameras 15-16 are held out "
            "for test; cameras 01-14 form the train/val pool. Probes generalization "
            "to unseen cameras of the *same* scenes."
        ),
        "v4.2": (
            "Scenario-disjoint: scenarios 18, 19, 42, 43 (all 16 cameras each) are "
            "held out for test; the remaining 18 scenarios form the train/val pool. "
            "Probes generalization to unseen scenes/scenarios."
        ),
    }[version]
    ps = stats["per_split"]
    total_imgs = sum(s["images"] for s in ps.values())
    total_boxes = sum(s["boxes"] for s in ps.values())
    rows = "\n".join(
        f"| {s:<5} | {ps[s]['images']:>7,} | {ps[s]['boxes']:>9,} | {ps[s]['negatives']:>10,} |"
        for s in SPLITS
    )
    readme = f"""# persondet_{version}

Single-class **person** detection dataset curated from the NIA **MTMDC**
multi-sensor trajectory-tracking source (`06_multisensor_trajectory_tracking`):
22 scenarios x 16 fixed cameras (1920x1080 @30fps), 2 camera layouts
(Space A indoor logistics: scenarios 01,10-19; Space B outdoor plaza: 31-39,42-43).

## Split strategy

{split_desc}

The train/val pool is split {int((1 - mtmdc.VAL_FRACTION) * 100)}/{int(mtmdc.VAL_FRACTION * 100)}
by a deterministic seeded shuffle (seed={mtmdc.SEED}).

## At a glance

| split | images | boxes | negatives |
|-------|--------:|----------:|-----------:|
{rows}

Total: **{total_imgs:,} images**, **{total_boxes:,} person boxes**.

## Class mapping

- One class only: `person` -> YOLO id `0`, COCO category `{{"id": 1, "name": "person"}}`.

## Layout

```
persondet_{version}/
├── data.yaml                                   # Ultralytics YOLO config
├── images/{{train,val,test}}/*.jpg               # hardlinked from the shared frame pool
├── labels/{{train,val,test}}/*.txt               # YOLO `0 cx cy w h` (empty = negative)
├── annotations/instances_{{train,val,test}}.json # COCO: bbox=[x,y,w,h] px, iscrowd=0
└── rfdetr/{{train,valid,test}}/                   # RF-DETR per-folder COCO layout
        ├── _annotations.coco.json
        └── *.jpg
```

## Usage

```bash
# YOLO
yolo detect train data=persondet_{version}/data.yaml model=yolo26n.pt
# RF-DETR: point a config's data.dataset_dir at persondet_{version}/rfdetr
```

## Provenance & processing

- **Source**: NIA MTMDC multi-sensor trajectory tracking (`06_multisensor_trajectory_tracking`).
  Per-frame JSON labels -> per-camera VATIC `.txt` (via the dataset's `json_to_vatic.py`);
  the VATIC `.txt` is the single source of truth for boxes here.
- **Frame subsampling**: every {stride}th annotated frame
  ({mtmdc.SOURCE_FPS}fps -> {mtmdc.SOURCE_FPS / stride:.2g}fps). For fixed cameras
  this temporal subsampling *is* the deduplication — no perceptual/CLIP dedup is
  run (curation.md sec.4: it would collapse a static sequence to ~1 frame). Frames
  are decoded sequentially (frame index == 0-based decode index, verified by box
  overlay) into a shared every-{mtmdc.SUBSAMPLE_EVERY}th-frame pool, then every
  {stride // mtmdc.SUBSAMPLE_EVERY}th pooled frame is hardlinked here.
- **Box cleaning**: all labels remapped to class `person`; boxes clamped to image
  bounds; boxes with width or height <= {mtmdc.MIN_BOX_PX:g}px dropped.
- **Negatives dropped**: frames with no labeled person are **excluded**. MTMDC
  annotates only the tracked-capture subjects, so ~50% of its zero-label frames
  contain real *unlabeled* bystanders (verified with a COCO person detector).
  Shipping them as background negatives would teach the detector to suppress real
  people (curation.md sec.5). Labeled frames are exhaustive (the GT has *more*
  boxes than a strong detector finds), so only labeled frames are kept.

## Caveats

- **Same source, different split** — v4.1 and v4.2 are built from the *same*
  extracted frames; do not pool their test sets, and never train on one and test
  on the other (the train pools overlap).
- Heavy occlusion is common (dense multi-person warehouse/plaza scenes); occluded
  persons are labelled and kept.
- Residual unlabeled bystanders: on peripheral/public-area cameras (e.g. parking
  entrances) a small fraction of kept frames may still contain an unlabeled
  passer-by. Empty frames (the worst offenders) are dropped; labeled frames are
  otherwise exhaustive for the controlled capture.
- Single domain (one capture campaign, two fixed-camera layouts) — combine with
  combined_v1/v2/v3 for broader coverage.
"""
    (version_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build(args.version, args.stride)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("version", choices=["v4.1", "v4.2"], help="Which version to build.")
    p.add_argument(
        "--stride",
        type=int,
        default=mtmdc.BUILD_STRIDE,
        help=f"Keep every Nth original frame; must be a multiple of the pool stride "
        f"({mtmdc.SUBSAMPLE_EVERY}). Default {mtmdc.BUILD_STRIDE} "
        f"({mtmdc.SOURCE_FPS // mtmdc.BUILD_STRIDE}fps).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
