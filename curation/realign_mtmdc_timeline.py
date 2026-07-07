"""Fix the MTMDC annotation-timeline misalignment; rebuild as a new version.

THE BUG (discovered 2026-07-03): MTMDC annotations (per-frame JSONs and the
VATIC derived from them) live on a **23 fps timeline**, while 13 of 16 cameras
per scenario record at **30 fps** (3 record at 23 fps). Ingestion
(`curation/mtmdc.py`) paired annotation index *i* with video frame *i*, which
is only correct for the 23 fps cameras. For 30 fps cameras the label lags the
image by a factor 30/23 — zero at frame 0, growing to ~75 s by sequence end
(video 9,600 frames vs 7,362 annotations covering the same 320 s). Training on
this teaches the model that moving people are background: the v5.1-L fine-tune
produces ZERO detections (conf>=0.25) on frames where stock finds 11 people,
and explains the MTMDC cam15/16 recall collapse of every fine-tuned model.

THE FIX: images on disk are correct video frames named by video index ``v``
(multiples of BUILD_STRIDE=30, so ``v * 23 / 30`` is an exact integer). Only
labels need remapping: for 30 fps cameras take annotation index
``v * 23 // 30``; for 23 fps cameras keep index ``v``. Per-camera fps is
probed from the source videos with ffprobe and cached.

Images whose corrected annotation has zero boxes are DROPPED (mirrors the
ingestion's KEEP_NEGATIVES=False: ~50% of empty-annotation frames contain real
unlabeled bystanders — keeping them would train person-suppression).

Builds ``persondet_<out>`` from ``persondet_<src>``: non-MTMDC files copied
verbatim (labels) / hardlinked (images); MTMDC labels re-derived from VATIC on
the corrected timeline; COCO + RF-DETR layouts rebuilt per split.

Usage (from the repo root):
    python -m curation.realign_mtmdc_timeline --src v6.1 --out v6.2
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess

from PIL import Image

from curation import mtmdc

logger = logging.getLogger("mtmdc.realign")

ANNOTATION_FPS = 23  # the uniform annotation timeline (TRACE registry: annotation_fps)
FPS_CACHE = mtmdc.DATA_ROOT / "_mtmdc_camera_fps.json"


def probe_camera_fps() -> dict[str, int]:
    """Probe (and cache) the integer fps of every scenario/camera source video.

    Returns:
        Mapping ``"s<NN>_c<NN>" -> fps`` (30 or 23).

    Raises:
        ValueError: If ffprobe reports an unexpected frame rate.
    """
    if FPS_CACHE.exists():
        return json.loads(FPS_CACHE.read_text())
    fps_map: dict[str, int] = {}
    for scenario in mtmdc.ALL_SCENARIOS:
        for camera in range(1, 17):
            video = mtmdc.camera_video(scenario, camera)
            if not video.exists():
                continue
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "csv=p=0",
                    str(video),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            num, _, den = out.partition("/")
            fps = round(int(num) / int(den or 1))
            if fps not in (23, 30):
                raise ValueError(f"unexpected fps {out} for {video}")
            fps_map[f"s{scenario:02d}_c{camera:02d}"] = fps
    if not fps_map:
        raise FileNotFoundError(
            f"no MTMDC source videos found under {mtmdc.RAW_ROOT} — is the NAS "
            "mounted? (nothing cached; fix the mount or RAW_ROOT and rerun)"
        )
    mtmdc.write_json(FPS_CACHE, fps_map)
    logger.info("probed %d cameras -> %s", len(fps_map), FPS_CACHE)
    return fps_map


def corrected_boxes(
    vatic: dict[int, list[mtmdc.Box]],
    video_frame: int,
    fps: int,
    width: int,
    height: int,
) -> list[mtmdc.Box]:
    """Annotation boxes for ``video_frame`` on the corrected timeline."""
    ann_idx = video_frame * ANNOTATION_FPS // fps if fps != ANNOTATION_FPS else video_frame
    return mtmdc.clean_boxes(vatic.get(ann_idx, []), width, height)


def build(src_version: str, out_version: str) -> None:
    """Build ``persondet_<out_version>`` with realigned MTMDC labels.

    Args:
        src_version: Base version (its non-MTMDC content is preserved verbatim).
        out_version: Output version id.

    Raises:
        KeyError: If an MTMDC image's camera has no probed fps (fail loud).
    """
    src = mtmdc.DATA_ROOT / f"persondet_{src_version}"
    out = mtmdc.DATA_ROOT / f"persondet_{out_version}"
    fps_map = probe_camera_fps()
    vatic_cache: dict[tuple[int, int], dict[int, list[mtmdc.Box]]] = {}

    stats: dict[str, dict[str, int]] = {}
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

        coco_images: list[dict] = []
        coco_anns: list[dict] = []
        n_mtmdc = n_dropped = n_other = n_boxes_before = n_boxes_after = 0
        for img_path in sorted((src / "images" / split).glob("*.jpg")):
            stem = img_path.stem
            with Image.open(img_path) as im:
                width, height = im.size
            if stem.startswith("mtmdc_"):
                scenario, camera, frame = mtmdc.parse_stem(stem)
                fps = fps_map.get(f"s{scenario:02d}_c{camera:02d}")
                if fps is None:
                    raise KeyError(f"no fps probed for {stem}")
                key = (scenario, camera)
                if key not in vatic_cache:
                    vatic_cache[key] = mtmdc.parse_vatic(mtmdc.camera_vatic(*key))
                old_label = (src / "labels" / split / f"{stem}.txt").read_text()
                n_boxes_before += sum(1 for line in old_label.splitlines() if line.strip())
                boxes = corrected_boxes(vatic_cache[key], frame, fps, width, height)
                if not boxes:
                    n_dropped += 1  # empty-after-remap: unlabeled-bystander risk
                    continue
                n_mtmdc += 1
                n_boxes_after += len(boxes)
                lines = []
                for b in boxes:
                    cx, cy, bw, bh = mtmdc.box_to_yolo(b, width, height)
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                text = "\n".join(lines) + "\n"
            else:
                text = (src / "labels" / split / f"{stem}.txt").read_text()
                boxes = None  # rebuilt below from text for COCO
                n_other += 1

            mtmdc.hardlink(img_path, out / "images" / split / img_path.name)
            mtmdc.hardlink(img_path, out / "rfdetr" / mtmdc.RFDETR_SPLIT_DIR[split] / img_path.name)
            (out / "labels" / split / f"{stem}.txt").write_text(text)

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
        mtmdc.write_json(
            out / "rfdetr" / mtmdc.RFDETR_SPLIT_DIR[split] / "_annotations.coco.json", coco
        )
        stats[split] = {
            "mtmdc_kept": n_mtmdc,
            "mtmdc_dropped_empty": n_dropped,
            "other": n_other,
            "mtmdc_boxes_before": n_boxes_before,
            "mtmdc_boxes_after": n_boxes_after,
        }
        logger.info("%s: %s", split, stats[split])

    (out / "data.yaml").write_text(
        f"# persondet_{out_version} — {src_version} with MTMDC labels realigned to the\n"
        f"# correct 23fps annotation timeline (30fps cameras were offset by x30/23).\n"
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "nc: 1\nnames: ['person']\n"
    )
    rows = "\n".join(
        f"| {s} | {t['mtmdc_kept']} | {t['mtmdc_dropped_empty']} | {t['other']} "
        f"| {t['mtmdc_boxes_before']} | {t['mtmdc_boxes_after']} |"
        for s, t in stats.items()
    )
    (out / "README.md").write_text(
        f"""# persondet_{out_version} — {src_version} + MTMDC timeline realignment

MTMDC annotations are on a 23 fps timeline; 13/16 cameras record at 30 fps.
The original ingestion paired annotation index i with video frame i, so labels
drifted by x30/23 through every 30 fps sequence (~75 s off at the end) —
training taught the model that moving people are background. This version
re-derives every MTMDC label from VATIC at index `video_frame * 23 / 30`
(exact integers; identity for the 23 fps cameras), drops images whose corrected
annotation is empty (unlabeled-bystander risk), and rebuilds COCO/RF-DETR.
Non-MTMDC sources (CrowdHuman-vbox `ch_`, MOT20 `mot_`, negatives `hn_oi_`)
are unchanged from {src_version}. See `curation/realign_mtmdc_timeline.py`.

| split | mtmdc kept | mtmdc dropped (empty) | other | mtmdc boxes before | after |
|---|--:|--:|--:|--:|--:|
{rows}

Known limitation: `persondet_v4.x` and every v5.x/v6.1 mix inherit the
misalignment; evaluate historical models accordingly.
""",
        encoding="utf-8",
    )
    logger.info("built persondet_%s", out_version)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", default="v6.1", help="Source version id (default v6.1).")
    p.add_argument("--out", default="v6.2", help="Output version id (default v6.2).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.src == args.out:
        raise ValueError("--src and --out must differ")
    build(args.src, args.out)


if __name__ == "__main__":
    main()
