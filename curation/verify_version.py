"""Stage 4 — verify a curated version's integrity (curation.md sec.6).

Independently re-reads a built ``persondet_v4.x`` from disk and asserts:

  * parity        — #images == #labels == #COCO images in every split;
  * no orphans    — every image has a label stem and vice-versa;
  * zero leakage  — no filename in two splits; and the version's holdout is
                    disjoint (cameras 15-16 for v4.1, scenarios 18/19/42/43 for
                    v4.2 never leak into train/val);
  * class sanity  — only YOLO class id 0;
  * box sanity    — 5 fields/line, all normalised coords in [0,1];
  * COCO match    — COCO image file_names == images on disk, bboxes in-bounds,
                    category_id == 1;
  * RF-DETR       — each rfdetr/{train,valid,test}/ has its images + COCO;
  * round-trip    — a sampled label re-derived from the source VATIC matches.

Exits non-zero if any check fails.

Usage (from the repo root):
    python -m curation.verify_version v4.1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from curation import mtmdc
from curation.build_version import RFDETR_SPLIT_DIR, SPLITS

logger = logging.getLogger("mtmdc.verify")

ROUNDTRIP_SAMPLES = 25  # labels re-derived from VATIC per split


class Checker:
    """Collects pass/fail results for one dataset."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def check(self, ok: bool, msg: str) -> None:
        if ok:
            self.passed += 1
        else:
            self.failures.append(msg)
            logger.error("FAIL: %s", msg)


def _stems(directory: Path, suffix: str) -> set[str]:
    return {p.stem for p in directory.glob(f"*{suffix}")}


def verify_split(version_dir: Path, split: str, chk: Checker) -> set[str]:
    """Run per-split checks; return the set of image stems in this split."""
    img_dir = version_dir / "images" / split
    lbl_dir = version_dir / "labels" / split
    coco_path = version_dir / "annotations" / f"instances_{split}.json"

    img_stems = _stems(img_dir, ".jpg")
    lbl_stems = _stems(lbl_dir, ".txt")
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco_stems = {Path(im["file_name"]).stem for im in coco["images"]}

    # Parity + orphans.
    chk.check(
        len(img_stems) == len(lbl_stems) == len(coco["images"]),
        f"[{split}] parity: images={len(img_stems)} labels={len(lbl_stems)} "
        f"coco={len(coco['images'])}",
    )
    chk.check(
        img_stems == lbl_stems,
        f"[{split}] image/label stems differ (orphans: {len(img_stems ^ lbl_stems)})",
    )
    chk.check(
        img_stems == coco_stems,
        f"[{split}] image/COCO stems differ (diff: {len(img_stems ^ coco_stems)})",
    )

    # COCO category + bbox bounds.
    cats = {c["id"] for c in coco["categories"]}
    chk.check(cats == {1}, f"[{split}] COCO categories != {{1}}: {cats}")
    sizes = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    bad_cat = bad_box = 0
    for a in coco["annotations"]:
        if a["category_id"] != 1:
            bad_cat += 1
        x, y, w, h = a["bbox"]
        iw, ih = sizes[a["image_id"]]
        if x < 0 or y < 0 or x + w > iw + 1 or y + h > ih + 1 or w <= 0 or h <= 0:
            bad_box += 1
    chk.check(bad_cat == 0, f"[{split}] {bad_cat} COCO anns with category_id != 1")
    chk.check(bad_box == 0, f"[{split}] {bad_box} COCO bboxes out of bounds/degenerate")

    # YOLO label sanity.
    bad_lines = bad_class = bad_coord = empty_labels = 0
    for txt in lbl_dir.glob("*.txt"):
        n_box = 0
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_box += 1
            parts = line.split()
            if len(parts) != 5:
                bad_lines += 1
                continue
            if parts[0] != "0":
                bad_class += 1
            if any(not (0.0 <= float(v) <= 1.0) for v in parts[1:]):
                bad_coord += 1
        if n_box == 0:
            empty_labels += 1
    chk.check(bad_lines == 0, f"[{split}] {bad_lines} YOLO lines without 5 fields")
    chk.check(bad_class == 0, f"[{split}] {bad_class} YOLO lines with class != 0")
    chk.check(bad_coord == 0, f"[{split}] {bad_coord} YOLO coords outside [0,1]")
    if not mtmdc.KEEP_NEGATIVES:
        chk.check(
            empty_labels == 0,
            f"[{split}] {empty_labels} empty labels present (negatives should be dropped)",
        )

    # RF-DETR per-folder layout.
    rf_dir = version_dir / "rfdetr" / RFDETR_SPLIT_DIR[split]
    rf_imgs = _stems(rf_dir, ".jpg")
    rf_coco = rf_dir / "_annotations.coco.json"
    chk.check(rf_imgs == img_stems, f"[{split}] rfdetr images != images/{split}")
    chk.check(rf_coco.is_file(), f"[{split}] missing {rf_coco}")

    return img_stems


def verify_leakage(version: str, per_split: dict[str, set[str]], chk: Checker) -> None:
    """No filename in two splits; version holdout disjoint from train/val."""
    train, val, test = per_split["train"], per_split["val"], per_split["test"]
    chk.check(not (train & val), f"{len(train & val)} stems shared train/val")
    chk.check(not (train & test), f"{len(train & test)} stems shared train/test")
    chk.check(not (val & test), f"{len(val & test)} stems shared val/test")

    def cams(stems: set[str]) -> set[int]:
        return {mtmdc.parse_stem(s)[1] for s in stems}

    def scns(stems: set[str]) -> set[int]:
        return {mtmdc.parse_stem(s)[0] for s in stems}

    pool = train | val
    if version == "v4.1":
        test_cams = cams(test)
        pool_cams = cams(pool)
        chk.check(
            test_cams <= mtmdc.V41_TEST_CAMERAS,
            f"v4.1 test cameras beyond {set(mtmdc.V41_TEST_CAMERAS)}: {test_cams}",
        )
        chk.check(
            not (pool_cams & mtmdc.V41_TEST_CAMERAS),
            f"v4.1 train/val leaks test cameras: {pool_cams & mtmdc.V41_TEST_CAMERAS}",
        )
    else:  # v4.2
        test_scns = scns(test)
        pool_scns = scns(pool)
        chk.check(
            test_scns <= mtmdc.V42_TEST_SCENARIOS,
            f"v4.2 test scenarios beyond {set(mtmdc.V42_TEST_SCENARIOS)}: {test_scns}",
        )
        chk.check(
            not (pool_scns & mtmdc.V42_TEST_SCENARIOS),
            f"v4.2 train/val leaks test scenarios: {pool_scns & mtmdc.V42_TEST_SCENARIOS}",
        )


def verify_roundtrip(version_dir: Path, chk: Checker) -> None:
    """Re-derive sampled YOLO labels straight from the source VATIC and compare."""
    rng = random.Random(mtmdc.SEED)
    mismatches = 0
    checked = 0
    for split in SPLITS:
        lbl_dir = version_dir / "labels" / split
        txts = sorted(lbl_dir.glob("*.txt"))
        if not txts:
            continue
        for txt in rng.sample(txts, min(ROUNDTRIP_SAMPLES, len(txts))):
            scenario, camera, frame = mtmdc.parse_stem(txt.stem)
            # Width/height from the COCO entry (authoritative per built image).
            vatic = mtmdc.parse_vatic(mtmdc.camera_vatic(scenario, camera))
            img = version_dir / "images" / split / f"{txt.stem}.jpg"
            import cv2  # local import: only needed for the round-trip read

            h, w = cv2.imread(str(img)).shape[:2]
            expected = mtmdc.clean_boxes(vatic.get(frame, []), w, h)
            written = [ln for ln in txt.read_text().splitlines() if ln.strip()]
            checked += 1
            if len(written) != len(expected):
                mismatches += 1
    chk.check(
        mismatches == 0, f"round-trip: {mismatches}/{checked} sampled labels disagree with VATIC"
    )


def verify(version: str) -> bool:
    version_dir = mtmdc.DATA_ROOT / f"persondet_{version}"
    if not version_dir.is_dir():
        raise FileNotFoundError(f"{version_dir} not found — build it first.")
    chk = Checker()
    per_split = {split: verify_split(version_dir, split, chk) for split in SPLITS}
    verify_leakage(version, per_split, chk)
    verify_roundtrip(version_dir, chk)

    total = sum(len(s) for s in per_split.values())
    if chk.failures:
        logger.error("%s: %d checks passed, %d FAILED", version, chk.passed, len(chk.failures))
        return False
    logger.info(
        "%s: all %d checks passed (%d images: train=%d val=%d test=%d)",
        version,
        chk.passed,
        total,
        len(per_split["train"]),
        len(per_split["val"]),
        len(per_split["test"]),
    )
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ok = verify(args.version)
    raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("version", choices=["v4.1", "v4.2"], help="Which version to verify.")
    return p.parse_args()


if __name__ == "__main__":
    main()
