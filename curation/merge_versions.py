"""Merge curated ``persondet`` versions into one combined training dataset.

curation.md sec.0: "Versions are independent datasets; they can later be merged
for training." This tool unions any number of already-curated versions (each in
the standard ``images|labels/{train,val,test}`` + ``annotations`` + ``rfdetr``
layout) into a new version, hardlinking images (shared bytes, one disk cost) and
re-indexing the COCO ids contiguously.

Natural merge only — every image of every source is kept (no balancing /
subsampling). Source filenames carry distinct prefixes (``ch_``, ``mot_``,
``mtmdc_`` …) so they never collide; this is asserted.

This is a *training* mix — evaluate per-source on the held-out splits that match
deployment (e.g. v4.1 test + WiseNET), not on a pooled test set.

Usage (from the repo root):
    python -m curation.merge_versions v5.1 --sources v2.1 v4.1
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from curation import mtmdc

logger = logging.getLogger("mtmdc.merge")


def _version_dir(version: str) -> Path:
    return mtmdc.DATA_ROOT / f"persondet_{version}"


def _reset_dirs(out_dir: Path) -> None:
    """Remove previously-merged outputs (never the source datasets)."""
    for sub in ("images", "labels", "annotations", "rfdetr"):
        target = out_dir / sub
        if target.exists():
            shutil.rmtree(target)
    for split in mtmdc.SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "rfdetr" / mtmdc.RFDETR_SPLIT_DIR[split]).mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)


def _merge_split(out_dir: Path, sources: list[Path], split: str) -> dict:
    """Merge one split across all sources; return per-source + total counts.

    Hardlinks each source image into ``images/<split>`` and ``rfdetr/<dir>``, its
    label into ``labels/<split>``, and concatenates the COCO with fresh ids.
    """
    merged: dict[str, list[dict]] = {
        "images": [],
        "annotations": [],
        "categories": [mtmdc.PERSON_COCO_CATEGORY],
    }
    img_id = ann_id = 0
    seen: set[str] = set()
    per_source: dict[str, dict[str, int]] = {}

    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    rf_dir = out_dir / "rfdetr" / mtmdc.RFDETR_SPLIT_DIR[split]

    for src in sources:
        coco = json.loads((src / "annotations" / f"instances_{split}.json").read_text())
        s_imgs = s_boxes = 0
        id_map: dict[int, int] = {}
        for im in coco["images"]:
            fname = im["file_name"]
            if fname in seen:
                raise ValueError(f"filename collision across sources in {split}: {fname}")
            seen.add(fname)
            src_img = src / "images" / split / fname
            src_lbl = src / "labels" / split / f"{Path(fname).stem}.txt"
            if not src_img.is_file() or not src_lbl.is_file():
                raise FileNotFoundError(f"missing image/label for {fname} in {src}/{split}")
            img_id += 1
            id_map[im["id"]] = img_id
            mtmdc.hardlink(src_img, img_dir / fname)
            mtmdc.hardlink(src_img, rf_dir / fname)
            mtmdc.hardlink(src_lbl, lbl_dir / f"{Path(fname).stem}.txt")
            merged["images"].append(
                {"id": img_id, "file_name": fname, "width": im["width"], "height": im["height"]}
            )
            s_imgs += 1
        for a in coco["annotations"]:
            ann_id += 1
            merged["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": id_map[a["image_id"]],
                    "category_id": mtmdc.PERSON_COCO_CATEGORY["id"],
                    "bbox": a["bbox"],
                    "area": a["area"],
                    "iscrowd": a.get("iscrowd", 0),
                }
            )
            s_boxes += 1
        per_source[src.name] = {"images": s_imgs, "boxes": s_boxes}

    mtmdc.write_json(out_dir / "annotations" / f"instances_{split}.json", merged)
    mtmdc.write_json(rf_dir / "_annotations.coco.json", merged)
    return {
        "images": len(merged["images"]),
        "boxes": len(merged["annotations"]),
        "per_source": per_source,
    }


def merge(version: str, source_versions: list[str]) -> dict:
    """Merge ``source_versions`` into ``persondet_<version>``; return stats."""
    out_dir = _version_dir(version)
    sources = [_version_dir(v) for v in source_versions]
    for src in sources:
        if not (src / "annotations").is_dir():
            raise FileNotFoundError(f"source dataset not found / not built: {src}")
    logger.info("merging %s -> %s", source_versions, out_dir.name)
    _reset_dirs(out_dir)

    per_split: dict[str, dict] = {}
    for split in mtmdc.SPLITS:
        sstats = _merge_split(out_dir, sources, split)
        per_split[split] = sstats
        logger.info(
            "  %-5s: %d images, %d boxes (%s)",
            split,
            sstats["images"],
            sstats["boxes"],
            ", ".join(f"{k}:{v['images']}img" for k, v in sstats["per_source"].items()),
        )

    _write_data_yaml(out_dir)
    _write_readme(out_dir, version, source_versions, per_split)
    logger.info("%s built at %s", version, out_dir)
    return {"version": version, "sources": source_versions, "per_split": per_split}


def _write_data_yaml(out_dir: Path) -> None:
    content = (
        f"# {out_dir.name} — merged training dataset ({out_dir.name})\n"
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names: ['person']\n"
    )
    (out_dir / "data.yaml").write_text(content, encoding="utf-8")


def _write_readme(
    out_dir: Path, version: str, source_versions: list[str], ps: dict[str, dict]
) -> None:
    total_imgs = sum(s["images"] for s in ps.values())
    total_boxes = sum(s["boxes"] for s in ps.values())
    split_rows = "\n".join(
        f"| {split:<5} | {ps[split]['images']:>8,} | {ps[split]['boxes']:>10,} |"
        for split in mtmdc.SPLITS
    )
    # Per-source contribution over all splits.
    src_tot: dict[str, dict[str, int]] = {}
    for s in ps.values():
        for name, c in s["per_source"].items():
            d = src_tot.setdefault(name, {"images": 0, "boxes": 0})
            d["images"] += c["images"]
            d["boxes"] += c["boxes"]
    src_rows = "\n".join(
        f"| {name} | {c['images']:>8,} | {c['boxes']:>10,} |" for name, c in src_tot.items()
    )
    readme = f"""# persondet_{version}

Merged single-class **person** detection dataset — a natural union (no
subsampling) of: {", ".join(source_versions)}.

Built for **training** a person detector. Evaluate on the held-out splits that
match deployment (e.g. `persondet_v4.1` test for unseen indoor cameras, and
WiseNET for an unseen building) — **not** on this dataset's pooled test split.

## At a glance

| split | images | boxes |
|-------|--------:|-----------:|
{split_rows}

Total: **{total_imgs:,} images**, **{total_boxes:,} person boxes**.

## Per-source contribution (all splits)

| source | images | boxes |
|--------|--------:|-----------:|
{src_rows}

Note the merge is **balanced by person instances**, not images: a crowd-dense
source (e.g. CrowdHuman in v2.1, ~27 boxes/img) contributes far more boxes per
image than fixed-camera video (v4.1, ~8 boxes/img), so a smaller image share can
still be ~half the training instances.

## Layout & class mapping

Standard layout: `images|labels/{{train,val,test}}`,
`annotations/instances_{{train,val,test}}.json` (COCO), and
`rfdetr/{{train,valid,test}}/` (RF-DETR per-folder COCO). One class: `person` ->
YOLO id `0`, COCO category `{{"id": 1, "name": "person"}}`. Images are hardlinked
from the source datasets (shared bytes, one disk cost).

## Usage

```bash
# YOLO
yolo detect train data=persondet_{version}/data.yaml model=yolo26n.pt
# RF-DETR: point a config's data.dataset_dir at persondet_{version}/rfdetr
```

## Caveats

- **Mixed domains & box conventions** inherited from the sources — see each
  source's own README for provenance, licensing, and convention notes.
- **Licensing is the union of the sources' licenses.** If any source is
  non-commercial (e.g. v2.1 = CrowdHuman + MOT20, CC BY-NC), this merged set is
  non-commercial too.
- This is a **training** mix; its `test` split pools domains and is **not** the
  deployment-representative evaluation (use the per-source held-out sets).
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    merge(args.version, args.sources)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("version", help="Output version id, e.g. v5.1 (-> persondet_v5.1).")
    p.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source version ids to merge, e.g. v2.1 v4.1 (-> persondet_v2.1, persondet_v4.1).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
