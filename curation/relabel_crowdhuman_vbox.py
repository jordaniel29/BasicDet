"""Rebuild the CrowdHuman labels of a curated version with visible boxes (vbox).

``persondet_v2.1`` ingested CrowdHuman with the amodal full-body box (``fbox``,
curation.md sec.3), which trains the detector to predict a person's full extent
*through* occluders. Our evaluation GT (CAMPUS VATIC, MTMDC) annotates only the
visible region, so on heavily occluded scenes (seated auditorium crowds) the
amodal predictions fall below the IoU-0.5 match threshold and score as FP + FN
simultaneously — the dominant cause of the CAMPUS-auditorium IDF1 collapse and
of full-body+part double detections (diagnosed 2026-07-03: fine-tuned nano
nested-pair rate 89/1k boxes vs stock 6.4/1k on auditorium).

This tool builds ``persondet_v2.2`` from ``persondet_v2.1``:

  * identical images and split membership (hardlinked, zero extra disk);
  * ``ch_`` labels re-derived from the raw CrowdHuman ``.odgt`` using **vbox**
    (same filtering as v2.1: ``tag=="person"`` only, ``extra.ignore==1`` and
    ``mask`` dropped, clamp to bounds, drop degenerate boxes);
  * all other sources' labels (``mot_``) copied verbatim;
  * COCO rebuilt per split from the resulting YOLO labels.

The raw ``.odgt`` files live in ``assets/data/_crowdhuman_odgt/`` (downloaded
from the official mirror https://huggingface.co/datasets/sshao0516/CrowdHuman;
box-format reference: https://arxiv.org/pdf/1805.00123).

Usage (from the repo root):
    python -m curation.relabel_crowdhuman_vbox            # v2.1 -> v2.2
    python -m curation.merge_versions v6.1 --sources v2.2 v4.1 hardneg_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from PIL import Image

from curation import mtmdc

logger = logging.getLogger("mtmdc.chvbox")

ODGT_DIR = mtmdc.DATA_ROOT / "_crowdhuman_odgt"
ODGT_FILES = ("annotation_train.odgt", "annotation_val.odgt")
CH_PREFIX = "ch_"


def load_odgt() -> dict[str, list[dict]]:
    """Map CrowdHuman image ID -> ``gtboxes`` records (train + val files).

    Returns:
        Dict keyed by the odgt ``ID`` (e.g. ``"273271,c9db000d5581f999"``).

    Raises:
        FileNotFoundError: If an odgt annotation file is missing.
    """
    records: dict[str, list[dict]] = {}
    for fname in ODGT_FILES:
        path = ODGT_DIR / fname
        if not path.is_file():
            raise FileNotFoundError(f"CrowdHuman odgt not found: {path}")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                records[rec["ID"]] = rec.get("gtboxes", [])
    logger.info("loaded %d odgt records from %s", len(records), ODGT_DIR)
    return records


def visible_boxes(gtboxes: list[dict], width: int, height: int) -> tuple[list[mtmdc.Box], int]:
    """Extract cleaned vbox pixel-corner boxes for annotated persons.

    Mirrors the v2.1 fbox ingestion rules with vbox geometry: keep
    ``tag=="person"``, skip ``extra.ignore==1`` (and ``mask`` rows via the tag
    check); clamp to image bounds; drop boxes with a side <= 1 px. Falls back
    to ``fbox`` when a record lacks ``vbox`` (rare in the official odgt).

    Args:
        gtboxes: The odgt ``gtboxes`` list for one image.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple of (cleaned pixel-corner boxes, count of fbox fallbacks).
    """
    raw: list[mtmdc.Box] = []
    n_fallback = 0
    for g in gtboxes:
        if g.get("tag") != "person":
            continue
        if g.get("extra", {}).get("ignore") == 1:
            continue
        box = g.get("vbox")
        if box is None:
            box = g.get("fbox")
            if box is None:
                continue
            n_fallback += 1
        x, y, w, h = (float(v) for v in box)
        raw.append((x, y, x + w, y + h))
    return mtmdc.clean_boxes(raw, width, height), n_fallback


def _yolo_lines(boxes: list[mtmdc.Box], width: int, height: int) -> str:
    lines = []
    for b in boxes:
        cx, cy, bw, bh = mtmdc.box_to_yolo(b, width, height)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _boxes_from_yolo(label_path: Path, width: int, height: int) -> list[mtmdc.Box]:
    """Denormalise an existing YOLO label file back to pixel corners (for COCO)."""
    boxes: list[mtmdc.Box] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts)
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        boxes.append((x1, y1, x1 + bw * width, y1 + bh * height))
    return boxes


def build(src_version: str, out_version: str) -> None:
    """Build ``persondet_<out_version>`` from ``persondet_<src_version>``.

    Args:
        src_version: Source version id whose images/splits are preserved.
        out_version: Output version id (directory must not equal the source).

    Raises:
        KeyError: If a ``ch_`` image has no matching odgt record (fail loud —
            a partial relabel would silently mix conventions).
    """
    src = mtmdc.DATA_ROOT / f"persondet_{src_version}"
    out = mtmdc.DATA_ROOT / f"persondet_{out_version}"
    odgt = load_odgt()

    totals: dict[str, dict[str, int]] = {}
    for split in mtmdc.SPLITS:
        for sub in ("images", "labels"):
            d = out / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        (out / "annotations").mkdir(exist_ok=True)

        coco_images: list[dict] = []
        coco_anns: list[dict] = []
        n_ch = n_other = n_boxes_ch = n_fallback = 0
        for img_path in sorted((src / "images" / split).glob("*.jpg")):
            stem = img_path.stem
            mtmdc.hardlink(img_path, out / "images" / split / img_path.name)
            width, height = Image.open(img_path).size
            label_out = out / "labels" / split / f"{stem}.txt"

            if stem.startswith(CH_PREFIX):
                ch_id = stem[len(CH_PREFIX) :]
                if ch_id not in odgt:
                    raise KeyError(f"{split}/{stem}: no odgt record for ID '{ch_id}'")
                boxes, fb = visible_boxes(odgt[ch_id], width, height)
                label_out.write_text(_yolo_lines(boxes, width, height))
                n_ch += 1
                n_boxes_ch += len(boxes)
                n_fallback += fb
            else:
                shutil.copy2(src / "labels" / split / f"{stem}.txt", label_out)
                boxes = _boxes_from_yolo(label_out, width, height)
                n_other += 1

            img_id = len(coco_images) + 1
            coco_images.append(
                {"id": img_id, "file_name": img_path.name, "width": width, "height": height}
            )
            for b in boxes:
                bbox, area = mtmdc.box_to_coco(b)
                coco_anns.append(
                    {
                        "id": len(coco_anns) + 1,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }
                )
        mtmdc.write_json(
            out / "annotations" / f"instances_{split}.json",
            {
                "images": coco_images,
                "annotations": coco_anns,
                "categories": [mtmdc.PERSON_COCO_CATEGORY],
            },
        )
        totals[split] = {
            "ch_images": n_ch,
            "other_images": n_other,
            "ch_boxes_vbox": n_boxes_ch,
            "fbox_fallbacks": n_fallback,
        }
        logger.info("%s: %s", split, totals[split])

    (out / "data.yaml").write_text(
        f"# persondet_{out_version} — v2.1 images/splits with CrowdHuman relabelled to vbox.\n"
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "nc: 1\nnames: ['person']\n"
    )
    (out / "README.md").write_text(_readme(src_version, out_version, totals), encoding="utf-8")
    logger.info("built persondet_%s at %s", out_version, out)


def _readme(src_version: str, out_version: str, totals: dict[str, dict[str, int]]) -> str:
    rows = "\n".join(
        f"| {split} | {t['ch_images']} | {t['other_images']} | {t['ch_boxes_vbox']} "
        f"| {t['fbox_fallbacks']} |"
        for split, t in totals.items()
    )
    return f"""# persondet_{out_version} — CrowdHuman relabelled to visible boxes (vbox)

Derived from **persondet_{src_version}**: identical images and split membership
(hardlinked), MOT20 labels verbatim, **CrowdHuman labels re-derived from the raw
`.odgt` using `vbox`** (visible region) instead of the amodal `fbox`.

**Why:** amodal fbox training makes the detector predict full-body extents
through occluders; visible-box benchmarks (CAMPUS VATIC, MTMDC) then score each
such prediction as FP + FN at IoU 0.5 (2026-07-03 auditorium diagnosis). See
`curation/relabel_crowdhuman_vbox.py`.

| split | ch_ images | other images | ch_ vbox boxes | fbox fallbacks |
|---|---|---|---|---|
{rows}

Filtering identical to v2.1: `tag=="person"` only; `extra.ignore==1` and `mask`
dropped; clamp to bounds; drop sides <= 1 px. License unchanged (**CC BY-NC** —
non-commercial, inherited from CrowdHuman/MOT20).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", default="v2.1", help="Source version id (default v2.1).")
    p.add_argument("--out", default="v2.2", help="Output version id (default v2.2).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.src == args.out:
        raise ValueError("--src and --out must differ")
    build(args.src, args.out)


if __name__ == "__main__":
    main()
