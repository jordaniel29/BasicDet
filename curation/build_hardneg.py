"""Package reviewed hard-negative images into a standard-layout negatives version.

Collects human-reviewed, person-free confuser images — the Open Images set
(`_oi_hardneg_staging/_verified_clean/`) plus the earlier `hardneg_v1` — into
`persondet_hardneg_v2`: single-class `person` layout with **every label empty**
(background images), negatives in `train` only. Merge into a training version:

    python -m curation.build_hardneg --out hardneg_v2
    python -m curation.merge_versions v5.3 --sources v5.2 hardneg_v2
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from PIL import Image

from curation import mtmdc

logger = logging.getLogger("mtmdc.hardneg")

OI_CLEAN = mtmdc.DATA_ROOT / "_oi_hardneg_staging" / "_verified_clean"
HARDNEG_V1_TRAIN = mtmdc.DATA_ROOT / "persondet_hardneg_v1" / "images" / "train"


def collect(oi_only: bool = False) -> list[Path]:
    """Gather all reviewed person-free negative images (Open Images + hardneg_v1).

    Args:
        oi_only: Exclude the hardneg_v1 frames. Those 11 images were cut from
            evaluation scenes (CAMPUS garden1/garden2/parkinglot + MTMDC
            camera16), so including them leaks eval backgrounds into training
            (2026-07-03 leakage audit).
    """
    negs = sorted(OI_CLEAN.glob("*/*.jpg"))
    if not oi_only:
        negs += sorted(HARDNEG_V1_TRAIN.glob("*.jpg"))
    by_name: dict[str, Path] = {}
    for p in negs:
        if p.name in by_name:
            raise ValueError(f"duplicate negative filename: {p.name}")
        by_name[p.name] = p
    return list(by_name.values())


def build(out: str, oi_only: bool = False) -> None:
    out_dir = mtmdc.DATA_ROOT / f"persondet_{out}"
    for split in mtmdc.SPLITS:
        for sub in ("images", "labels"):
            d = out_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    negs = collect(oi_only=oi_only)
    images: list[dict] = []
    for i, src in enumerate(negs, start=1):
        mtmdc.hardlink(src, out_dir / "images" / "train" / src.name)
        (out_dir / "labels" / "train" / f"{src.stem}.txt").write_text("")  # empty = background
        w, h = Image.open(src).size
        images.append({"id": i, "file_name": src.name, "width": w, "height": h})

    cats = [mtmdc.PERSON_COCO_CATEGORY]
    mtmdc.write_json(
        out_dir / "annotations" / "instances_train.json",
        {"images": images, "annotations": [], "categories": cats},
    )
    for split in ("val", "test"):
        mtmdc.write_json(
            out_dir / "annotations" / f"instances_{split}.json",
            {"images": [], "annotations": [], "categories": cats},
        )
    (out_dir / "data.yaml").write_text(
        f"# persondet_{out} — hard-negative background images (person-free confusers)\n"
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "nc: 1\nnames: ['person']\n"
    )
    logger.info("built persondet_%s: %d person-free negatives (train only)", out, len(images))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="hardneg_v2", help="Output negatives version name.")
    p.add_argument(
        "--oi-only",
        action="store_true",
        help="Exclude the hardneg_v1 eval-scene frames (leak-free pack, e.g. hardneg_v3).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    build(args.out, oi_only=args.oi_only)


if __name__ == "__main__":
    main()
