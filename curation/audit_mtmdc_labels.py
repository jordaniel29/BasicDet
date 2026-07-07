"""Audit MTMDC labels for missing (unannotated) people; densify into a new version.

Motivation (2026-07-03): ~50% of MTMDC's zero-label frames were found to contain
real unlabeled bystanders (`curation/mtmdc.py` KEEP_NEGATIVES note). Unlabeled
people inside *annotated* training frames act as implicit hard negatives — the
model is penalised for detecting them and learns to suppress people in
MTMDC-like scenes, a direct recall killer on 77% of the training images.

Method — cross-model consensus, cache-once / report-first / reversible:

  1. ``scan``   run two detectors over the MTMDC train images and cache every
                detection (conf >= 0.25) to JSONL. Detectors: the in-domain
                YOLO26-L (v5.1 fine-tune) and RF-DETR-Large fine-tuned on
                v2 (CrowdHuman+MOT20) — the latter has NEVER seen MTMDC labels,
                so it carries no bias from the very holes we are auditing.
  2. ``report`` from the caches: a candidate = a detection that matches no GT
                box (IoU < IOU_MATCH and no containment overlap — containment
                is excused because both detectors were fbox/amodal-trained
                while MTMDC GT is visible-region). A candidate survives only if
                BOTH models propose it (cross-model IoU >= CONSENSUS_IOU).
                Emits a threshold grid, a manifest CSV, and an HTML contact
                sheet of sampled crops for human review.
  3. ``apply``  build ``persondet_v6.2`` = v6.1 with the accepted pseudo-boxes
                appended to the affected *train* labels (val/test untouched —
                pseudo-labels in val would bias model selection toward the
                labelling models). COCO + RF-DETR layouts rebuilt for train,
                copied verbatim for val/test. Provenance in ``_densify/``.

Usage (from the repo root, conda env `persondet`, GPU via CUDA_VISIBLE_DEVICES):
    python -m curation.audit_mtmdc_labels scan
    python -m curation.audit_mtmdc_labels report --yolo-conf 0.6 --rfdetr-conf 0.6
    python -m curation.audit_mtmdc_labels apply --out v6.2 --yolo-conf 0.6 --rfdetr-conf 0.6
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from curation import mtmdc

logger = logging.getLogger("mtmdc.audit")

BASE_VERSION = "v6.1"
AUDIT_DIR = mtmdc.DATA_ROOT / "_mtmdc_label_audit"
YOLO_WEIGHTS = (
    Path.home() / "jordan/Person-Det/runs/detect/yolo26/yolo26l_person_v5.1/weights/best.pt"
)
RFDETR_WEIGHTS = (
    Path.home() / "jordan/Person-Det/runs/rfdetr/rfdetr_large_person_v2/checkpoint_best_ema.pth"
)
CACHE_CONF = 0.25  # cache floor — re-threshold later without re-inference
IOU_MATCH = 0.30  # pred with IoU >= this vs any GT box counts as "already labelled"
CONTAINMENT_MATCH = 0.60  # inter/min-area >= this also counts as matched (fbox vs vbox)
CONSENSUS_IOU = 0.50  # cross-model agreement required to survive
MIN_SIDE_PX = 6.0  # pseudo-boxes smaller than this are too unreliable to add


# --------------------------------------------------------------------------- #
# shared geometry / IO
# --------------------------------------------------------------------------- #
def _iou_and_containment(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (IoU, intersection-over-smaller-area) of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0), (inter / smaller if smaller > 0 else 0.0)


def _mtmdc_train_images() -> list[Path]:
    base = mtmdc.DATA_ROOT / f"persondet_{BASE_VERSION}" / "images" / "train"
    return sorted(base.glob("mtmdc_*.jpg"))


def _gt_boxes(image: Path) -> list[list[float]]:
    """GT xyxy pixel boxes from the base version's YOLO label."""
    label = image.parent.parent.parent / "labels" / "train" / f"{image.stem}.txt"
    with Image.open(image) as im:
        width, height = im.size
    boxes = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        boxes.append([x1, y1, x1 + bw * width, y1 + bh * height])
    return boxes


def _load_cache(path: Path) -> dict[str, list[list[float]]]:
    preds: dict[str, list[list[float]]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            preds[rec["img"]] = rec["boxes"]
    return preds


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def scan(batch: int, rfdetr_targets: Path | None = None) -> None:
    """Run both detectors over MTMDC train images, caching detections to JSONL.

    Resumable: images already present in a cache file are skipped, so an
    interrupted scan continues where it left off.

    Args:
        batch: YOLO inference batch size (RF-DETR runs single-image).
        rfdetr_targets: Optional file of image names (one per line) limiting the
            RF-DETR pass. RF-DETR at ~2 img/s cannot sweep all 57k frames in
            reasonable time, and it only needs to (a) confirm/refute YOLO's
            candidate frames for the consensus gate and (b) cover an unbiased
            random sample to measure what the MTMDC-blind model finds that the
            label-biased YOLO misses.
    """
    AUDIT_DIR.mkdir(exist_ok=True)
    images = _mtmdc_train_images()
    logger.info("%d MTMDC train images to scan", len(images))

    # ---- YOLO26-L ----
    yolo_cache = AUDIT_DIR / "yolo26l_preds.jsonl"
    done = set(_load_cache(yolo_cache))
    todo = [p for p in images if p.name not in done]
    logger.info("YOLO26-L: %d cached, %d to run", len(done), len(todo))
    if todo:
        from ultralytics import YOLO

        model = YOLO(str(YOLO_WEIGHTS))
        with open(yolo_cache, "a", encoding="utf-8") as fh:
            for i in range(0, len(todo), batch):
                chunk = todo[i : i + batch]
                results = model.predict(
                    [str(p) for p in chunk],
                    conf=CACHE_CONF,
                    imgsz=640,
                    device=0,
                    half=True,
                    verbose=False,
                )
                for p, r in zip(chunk, results, strict=True):
                    boxes = [
                        [round(float(v), 1) for v in xyxy] + [round(float(c), 4)]
                        for xyxy, c in zip(
                            r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), strict=True
                        )
                    ]
                    fh.write(json.dumps({"img": p.name, "boxes": boxes}) + "\n")
                if (i // batch) % 50 == 0:
                    fh.flush()
                    logger.info("YOLO26-L: %d/%d", min(i + batch, len(todo)), len(todo))

    # ---- RF-DETR-Large (v2 fine-tune; MTMDC-blind) ----
    rf_cache = AUDIT_DIR / "rfdetr_preds.jsonl"
    done = set(_load_cache(rf_cache))
    todo = [p for p in images if p.name not in done]
    if rfdetr_targets is not None:
        wanted = set(rfdetr_targets.read_text().split())
        todo = [p for p in todo if p.name in wanted]
    logger.info("RF-DETR-L: %d cached, %d to run", len(done), len(todo))
    if todo:
        from rfdetr import RFDETRLarge

        model = RFDETRLarge(pretrain_weights=str(RFDETR_WEIGHTS))
        try:
            model.optimize_for_inference()
            logger.info("RF-DETR optimized for inference")
        except Exception as exc:  # noqa: BLE001 — best-effort speedup, safe fallback
            logger.warning("optimize_for_inference failed (%s); continuing unoptimized", exc)
        with open(rf_cache, "a", encoding="utf-8") as fh:
            for i, p in enumerate(todo):
                with Image.open(p) as im:
                    det = model.predict(im.convert("RGB"), threshold=CACHE_CONF)
                boxes = [
                    [round(float(v), 1) for v in xyxy] + [round(float(c), 4)]
                    for xyxy, c in zip(det.xyxy, det.confidence, strict=True)
                ]
                fh.write(json.dumps({"img": p.name, "boxes": boxes}) + "\n")
                if i % 500 == 0:
                    fh.flush()
                    logger.info("RF-DETR-L: %d/%d", i, len(todo))
    logger.info("scan complete: caches in %s", AUDIT_DIR)


# --------------------------------------------------------------------------- #
# candidates (from caches; no GPU)
# --------------------------------------------------------------------------- #
def _unmatched(preds: list[list[float]], gt: list[list[float]], conf: float) -> list[list[float]]:
    out = []
    for b in preds:
        if b[4] < conf:
            continue
        if (b[2] - b[0]) < MIN_SIDE_PX or (b[3] - b[1]) < MIN_SIDE_PX:
            continue
        matched = False
        for g in gt:
            iou, cont = _iou_and_containment(b[:4], g)
            if iou >= IOU_MATCH or cont >= CONTAINMENT_MATCH:
                matched = True
                break
        if not matched:
            out.append(b)
    return out


def candidates(yolo_conf: float, rfdetr_conf: float) -> dict[str, list[dict]]:
    """Cross-model consensus candidates per image at the given thresholds.

    Returns:
        Mapping image name -> list of {"box": xyxy, "yolo_conf", "rfdetr_conf"}.
    """
    yolo = _load_cache(AUDIT_DIR / "yolo26l_preds.jsonl")
    rf = _load_cache(AUDIT_DIR / "rfdetr_preds.jsonl")
    images = {p.name: p for p in _mtmdc_train_images()}
    both = sorted(set(yolo) & set(rf) & set(images))
    result: dict[str, list[dict]] = {}
    for name in both:
        gt = _gt_boxes(images[name])
        y_cand = _unmatched(yolo[name], gt, yolo_conf)
        if not y_cand:
            continue
        r_cand = _unmatched(rf[name], gt, rfdetr_conf)
        if not r_cand:
            continue
        found = []
        for yb in y_cand:
            best = max(r_cand, key=lambda rb: _iou_and_containment(yb[:4], rb[:4])[0])
            if _iou_and_containment(yb[:4], best[:4])[0] >= CONSENSUS_IOU:
                found.append({"box": yb[:4], "yolo_conf": yb[4], "rfdetr_conf": best[4]})
        if found:
            result[name] = found
    return result


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def report(yolo_conf: float, rfdetr_conf: float, sheet_n: int) -> None:
    """Write threshold grid, manifest CSV, and an HTML contact sheet."""
    grid_rows = []
    for yc, rc in [(0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8)]:
        cand = candidates(yc, rc)
        n_boxes = sum(len(v) for v in cand.values())
        grid_rows.append((yc, rc, len(cand), n_boxes))
        logger.info(
            "thresholds y>=%.1f r>=%.1f -> %d images, %d candidate boxes",
            yc,
            rc,
            len(cand),
            n_boxes,
        )

    cand = candidates(yolo_conf, rfdetr_conf)
    per_cam: dict[str, int] = defaultdict(int)
    for name, boxes in cand.items():
        s, c, _ = mtmdc.parse_stem(Path(name).stem)
        per_cam[f"s{s:02d}_c{c:02d}"] += len(boxes)

    with open(AUDIT_DIR / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "x1", "y1", "x2", "y2", "yolo_conf", "rfdetr_conf"])
        for name in sorted(cand):
            for c in cand[name]:
                w.writerow(
                    [name] + [round(v, 1) for v in c["box"]] + [c["yolo_conf"], c["rfdetr_conf"]]
                )

    # contact sheet: sampled crops, GT boxes drawn for context
    sample = random.Random(42).sample(sorted(cand), min(sheet_n, len(cand)))
    images = {p.name: p for p in _mtmdc_train_images()}
    cells = []
    for name in sample:
        img = Image.open(images[name]).convert("RGB")
        for c in cand[name]:
            x1, y1, x2, y2 = c["box"]
            pad = 60
            crop = img.crop(
                (
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(img.width, x2 + pad),
                    min(img.height, y2 + pad),
                )
            )
            crop.thumbnail((320, 320))
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            cells.append(
                f'<div style="display:inline-block;margin:4px;text-align:center">'
                f'<img src="data:image/jpeg;base64,{b64}"><br>'
                f"<small>{name}<br>y={c['yolo_conf']:.2f} r={c['rfdetr_conf']:.2f}</small></div>"
            )
    grid_html = "".join(
        f"<tr><td>{yc}</td><td>{rc}</td><td>{ni}</td><td>{nb}</td></tr>"
        for yc, rc, ni, nb in grid_rows
    )
    cam_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(per_cam.items()))
    (AUDIT_DIR / "contact_sheet.html").write_text(
        f"<h1>MTMDC missing-label candidates (y>={yolo_conf}, r>={rfdetr_conf})</h1>"
        f"<h2>Threshold grid</h2><table border=1><tr><th>yolo</th><th>rfdetr</th>"
        f"<th>images</th><th>boxes</th></tr>{grid_html}</table>"
        f"<h2>Per camera</h2><table border=1>{cam_html}</table>"
        f"<h2>Sampled candidate crops (60px context)</h2>{cells and ''.join(cells)}",
        encoding="utf-8",
    )
    logger.info(
        "report: %d images / %d boxes at y>=%.2f r>=%.2f -> %s",
        len(cand),
        sum(len(v) for v in cand.values()),
        yolo_conf,
        rfdetr_conf,
        AUDIT_DIR / "contact_sheet.html",
    )


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def apply(out_version: str, yolo_conf: float, rfdetr_conf: float) -> None:
    """Build ``persondet_<out>`` = base version + accepted pseudo-boxes (train only)."""
    src = mtmdc.DATA_ROOT / f"persondet_{BASE_VERSION}"
    out = mtmdc.DATA_ROOT / f"persondet_{out_version}"
    cand = candidates(yolo_conf, rfdetr_conf)
    n_boxes = sum(len(v) for v in cand.values())
    logger.info("applying %d pseudo-boxes on %d images -> %s", n_boxes, len(cand), out)

    for split in mtmdc.SPLITS:
        for sub in ("images", "labels"):
            d = out / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        rf_dir = out / "rfdetr" / mtmdc.RFDETR_SPLIT_DIR[split]
        if rf_dir.exists():
            shutil.rmtree(rf_dir)
        rf_dir.mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(exist_ok=True)
    dens_dir = out / "_densify"
    dens_dir.mkdir(exist_ok=True)

    for split in mtmdc.SPLITS:
        coco_images: list[dict] = []
        coco_anns: list[dict] = []
        rf_dir = out / "rfdetr" / mtmdc.RFDETR_SPLIT_DIR[split]
        for img_path in sorted((src / "images" / split).glob("*.jpg")):
            mtmdc.hardlink(img_path, out / "images" / split / img_path.name)
            mtmdc.hardlink(img_path, rf_dir / img_path.name)
            with Image.open(img_path) as im:
                width, height = im.size
            label_src = src / "labels" / split / f"{img_path.stem}.txt"
            text = label_src.read_text()
            if split == "train" and img_path.name in cand:
                extra = []
                for c in cand[img_path.name]:
                    box = mtmdc.clamp_box(tuple(c["box"]), width, height)
                    if not mtmdc.is_valid_box(box):
                        continue
                    cx, cy, bw, bh = mtmdc.box_to_yolo(box, width, height)
                    extra.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if extra:
                    if text and not text.endswith("\n"):
                        text += "\n"
                    text += "\n".join(extra) + "\n"
            (out / "labels" / split / f"{img_path.stem}.txt").write_text(text)

            img_id = len(coco_images) + 1
            coco_images.append(
                {"id": img_id, "file_name": img_path.name, "width": width, "height": height}
            )
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                bbox, area = mtmdc.box_to_coco((x1, y1, x1 + bw * width, y1 + bh * height))
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
        coco = {
            "images": coco_images,
            "annotations": coco_anns,
            "categories": [mtmdc.PERSON_COCO_CATEGORY],
        }
        mtmdc.write_json(out / "annotations" / f"instances_{split}.json", coco)
        mtmdc.write_json(rf_dir / "_annotations.coco.json", coco)

    shutil.copy2(AUDIT_DIR / "manifest.csv", dens_dir / "manifest.csv")
    mtmdc.write_json(
        dens_dir / "params.json",
        {
            "base": BASE_VERSION,
            "yolo_conf": yolo_conf,
            "rfdetr_conf": rfdetr_conf,
            "iou_match": IOU_MATCH,
            "containment_match": CONTAINMENT_MATCH,
            "consensus_iou": CONSENSUS_IOU,
            "images_densified": len(cand),
            "pseudo_boxes": n_boxes,
            "yolo_weights": str(YOLO_WEIGHTS),
            "rfdetr_weights": str(RFDETR_WEIGHTS),
        },
    )
    (out / "data.yaml").write_text(
        f"# persondet_{out_version} — {BASE_VERSION} + MTMDC missing-label densification.\n"
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "nc: 1\nnames: ['person']\n"
    )
    (out / "README.md").write_text(
        f"# persondet_{out_version} — {BASE_VERSION} + MTMDC label densification\n\n"
        f"Identical to persondet_{BASE_VERSION} except **{n_boxes} pseudo-boxes** added to\n"
        f"{len(cand)} MTMDC **train** labels — unlabeled people recovered by cross-model\n"
        f"consensus (YOLO26-L v5.1 >= {yolo_conf} AND RF-DETR-L v2 >= {rfdetr_conf},\n"
        f"cross-model IoU >= {CONSENSUS_IOU}, no GT match at IoU {IOU_MATCH} /\n"
        f"containment {CONTAINMENT_MATCH}). val/test untouched. Audit trail in\n"
        f"`_densify/` (manifest + params). See `curation/audit_mtmdc_labels.py`.\n\n"
        f"Why: unlabeled people in training frames act as implicit hard negatives and\n"
        f"suppress recall on the dominant (77%) MTMDC domain.\n",
        encoding="utf-8",
    )
    logger.info("built persondet_%s", out_version)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="Run both detectors, cache detections (GPU).")
    s.add_argument("--batch", type=int, default=8, help="YOLO batch size (default 8).")
    s.add_argument(
        "--rfdetr-targets",
        type=Path,
        default=None,
        help="File of image names limiting the RF-DETR pass (see scan docstring).",
    )
    for name in ("report", "apply"):
        q = sub.add_parser(name)
        q.add_argument("--yolo-conf", type=float, default=0.6)
        q.add_argument("--rfdetr-conf", type=float, default=0.6)
        if name == "report":
            q.add_argument("--sheet-n", type=int, default=80, help="Contact-sheet crops.")
        else:
            q.add_argument("--out", default="v6.2", help="Output version id.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    match args.cmd:
        case "scan":
            scan(args.batch, args.rfdetr_targets)
        case "report":
            report(args.yolo_conf, args.rfdetr_conf, args.sheet_n)
        case "apply":
            apply(args.out, args.yolo_conf, args.rfdetr_conf)


if __name__ == "__main__":
    main()
