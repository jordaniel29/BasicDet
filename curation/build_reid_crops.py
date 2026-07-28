"""Build, review, and package the MTMDC ReID crop dataset.

Cuts identity-labelled person crops from the MTMDC frames using the VATIC
tracks on the CORRECTED annotation timeline (see
``curation/realign_mtmdc_timeline.py`` — crops cut on the raw timeline would be
floor tiles), then supports a human-in-the-loop review before packaging into
the Market-1501 layout consumed by ``basicdet``'s ReID trainers.

Three stages (cache-once / report-first / reversible — curation.md sec.4):

  1. ``build``   crop every sampled track box into ``_reid_staging/crops/``
                 (named ``<pid>_c<cam>_s<scenario>_f<frame>.jpg``) + a manifest.
                 Identity = (scenario, NIA global person id from the JSON ``attributes[].pid``) as
                 ``pid = scenario * 10000 + nia_pid`` — NIA pids are consistent across
                 cameras within a scenario (VATIC track_ids are NOT — per-camera only); actors
                 appearing in two scenarios become two identities (harmless for
                 training, noted in the README).
  2. ``review``  embed all crops (deployed CLIP-ReID by default) and emit
                 ``review/identity_report.html``: one row per identity, crops
                 grouped by camera, sorted by embedding coherence (worst
                 first), plus a suspected-duplicate-identity section and a
                 ``decisions.csv`` template (``keep`` / ``drop`` /
                 ``merge:<pid>``).
  3. ``package`` apply ``decisions.csv`` and write the Market-1501 layout
                 (``bounding_box_train/``, ``query/``, ``bounding_box_test/``)
                 with a scenario-disjoint identity split: held-out scenarios
                 provide the query/gallery identities (query = one camera per
                 identity, gallery = the rest).

Usage (from the repo root, conda env ``persondet``):
    python -m curation.build_reid_crops build
    python -m curation.build_reid_crops review            # then edit decisions.csv
    python -m curation.build_reid_crops package --out reid_v1
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from curation import mtmdc
from curation.realign_mtmdc_timeline import probe_camera_fps

logger = logging.getLogger("mtmdc.reid_crops")

STAGING = mtmdc.DATA_ROOT / "_reid_staging"
FRAME_SOURCE = mtmdc.DATA_ROOT / "persondet_v6.2"  # realigned; images keyed by video index

# Identity split: whole scenarios held out for query/gallery (mirrors v4.2's
# scenario holdout — one indoor pair + one outdoor pair).
HELDOUT_SCENARIOS = (18, 19, 42, 43)

SAMPLE_STRIDE_S = 2  # keep a crop every N seconds per (identity, camera)
MIN_CROP_H = 80  # px in the 1080p frame; smaller is useless at 256x128
MIN_CROP_W = 32
PAD_FRAC = 0.08  # context padding around the box
PID_SCENARIO_BASE = 10000  # pid = scenario * base + track_id

# TRACE deployment paths for the review embedder (config-free convenience;
# review still works without them via --embedder none).
TRACE_CLIPREID_PKG = Path("/home/jordan/jordan/TRACE_SSAVE-AI-MVP/packages/piaspace-clip-reid/src")
TRACE_CLIPREID_WEIGHTS = Path(
    "/home/jordan/jordan/TRACE_SSAVE-AI-MVP/weights/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth"
)
# The deployed TRT engine (preferred: the .pth is not kept on this box). Running
# it requires an env with tensorrt (conda `trace`) + piaspace-trt-runtime on
# PYTHONPATH.
TRACE_CLIPREID_ENGINE = Path(
    "/home/jordan/jordan/TRACE_SSAVE-AI-MVP/weights/clipreid_person.fp16.engine"
)


@dataclass(frozen=True)
class TrackBox:
    """One annotated person box on the corrected timeline.

    Attributes:
        pid: GLOBAL person id — the NIA JSON ``attributes[].pid``, consistent
            across cameras within a scenario. (The VATIC export only keeps the
            per-camera ``track_id``; keying identities on it mixes different
            people across cameras — caught by the coherence review 2026-07-13.)
        box: Pixel corners (x1, y1, x2, y2).
        occluded: NIA occluded flag.
    """

    pid: int
    box: mtmdc.Box
    occluded: bool


def parse_vatic_tracks(path: Path) -> dict[int, list[TrackBox]]:
    """Parse a VATIC ``.txt`` (frame -> boxes with PER-CAMERA track ids).

    .. warning:: VATIC ``track_id`` is per-camera, NOT a global identity —
        use :func:`json_person_boxes` for ReID. Kept for diagnostics only.

    Args:
        path: Per-camera VATIC file (columns: ``track_id x1 y1 x2 y2 frame
            lost occluded generated "label"``).

    Returns:
        Mapping from annotation-frame index to its boxes; ``lost`` rows dropped.
    """
    frames: dict[int, list[TrackBox]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 10 or parts[6] == "1":  # malformed or lost
                continue
            frames[int(float(parts[5]))].append(
                TrackBox(
                    pid=int(parts[0]),
                    box=(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])),
                    occluded=parts[7] == "1",
                )
            )
    return frames


def _camera_json_index(scenario: int, camera: int) -> dict[int, Path]:
    """Map annotation-frame index -> per-frame NIA JSON path for one camera."""
    out: dict[int, Path] = {}
    for p in mtmdc.camera_json_dir(scenario, camera).glob("*.json"):
        m = mtmdc.FRAME_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def json_person_boxes(path: Path) -> list[TrackBox]:
    """Extract person boxes with GLOBAL pids from one per-frame NIA JSON.

    Objects without a ``pid`` attribute or marked ``outside`` are skipped.

    Args:
        path: One ``NIA_MTMDC_*_<frame>.json`` file.

    Returns:
        The frame's person boxes with scenario-global person ids.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes: list[TrackBox] = []
    for obj in data.get("objects", []):
        if obj.get("label") != "person" or obj.get("outside") == "1":
            continue
        pid = None
        for attr in obj.get("attributes", []):
            if "pid" in attr:
                pid = int(attr["pid"])
                break
        if pid is None or not obj.get("position"):
            continue
        pos = obj["position"][0]
        x, y = float(pos["x"]), float(pos["y"])
        boxes.append(
            TrackBox(
                pid=pid,
                box=(x, y, x + float(pos["width"]), y + float(pos["height"])),
                occluded=obj.get("occluded") == "1",
            )
        )
    return boxes


def _frame_files() -> dict[tuple[int, int], list[tuple[int, Path]]]:
    """Index the realigned MTMDC images: (scenario, camera) -> [(frame, path)]."""
    out: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)
    for split in mtmdc.SPLITS:
        for p in (FRAME_SOURCE / "images" / split).glob("mtmdc_*.jpg"):
            scenario, camera, frame = mtmdc.parse_stem(p.stem)
            out[(scenario, camera)].append((frame, p))
    for frames in out.values():
        frames.sort()
    return out


def build() -> None:
    """Stage 1: cut sampled track crops into the staging area + manifest."""
    fps_map = probe_camera_fps()
    frames = _frame_files()
    crops_dir = STAGING / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = STAGING / "manifest.csv"
    n_crops = 0
    identities: set[int] = set()
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "pid", "scenario", "camera", "frame", "occluded"])
        for scenario in mtmdc.ALL_SCENARIOS:
            for camera in range(1, 17):
                fps = fps_map.get(f"s{scenario:02d}_c{camera:02d}")
                if fps is None:
                    continue
                json_index = _camera_json_index(scenario, camera)
                if not json_index:
                    continue
                last_kept: dict[int, int] = {}  # global pid -> last kept video frame
                # Video frames on disk are multiples of BUILD_STRIDE (1 fps).
                for video_frame, img_path in frames.get((scenario, camera), []):
                    ann_idx = video_frame * 23 // fps if fps != 23 else video_frame
                    json_path = json_index.get(ann_idx)
                    if json_path is None:
                        continue
                    for tb in json_person_boxes(json_path):
                        if video_frame - last_kept.get(tb.pid, -(10**9)) < (SAMPLE_STRIDE_S * 30):
                            continue
                        x1, y1, x2, y2 = mtmdc.clamp_box(tb.box, 1920, 1080)
                        if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
                            continue
                        pad_w, pad_h = (x2 - x1) * PAD_FRAC, (y2 - y1) * PAD_FRAC
                        pid = scenario * PID_SCENARIO_BASE + tb.pid
                        name = f"{pid}_c{camera}_s{scenario:02d}_f{video_frame:06d}.jpg"
                        with Image.open(img_path) as im:
                            crop = im.crop(
                                (
                                    max(0, int(x1 - pad_w)),
                                    max(0, int(y1 - pad_h)),
                                    min(im.width, int(x2 + pad_w)),
                                    min(im.height, int(y2 + pad_h)),
                                )
                            )
                            crop.save(crops_dir / name, quality=95)
                        writer.writerow(
                            [name, pid, scenario, camera, video_frame, int(tb.occluded)]
                        )
                        last_kept[tb.pid] = video_frame
                        identities.add(pid)
                        n_crops += 1
            logger.info("scenario %02d done (%d crops so far)", scenario, n_crops)
    logger.info("built %d crops / %d identities -> %s", n_crops, len(identities), crops_dir)


def _load_manifest() -> list[dict[str, str]]:
    with open(STAGING / "manifest.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _clipreid_embedder(device: str) -> object:
    if str(TRACE_CLIPREID_PKG) not in sys.path:
        sys.path.insert(0, str(TRACE_CLIPREID_PKG))
    from piaspace_clip_reid import CLIPReIDEmbedder

    return CLIPReIDEmbedder(
        {
            "device": device,
            "input_size": [256, 128],
            "stride": 12,
            "engine_path": str(TRACE_CLIPREID_ENGINE),
            "weights_path": str(TRACE_CLIPREID_WEIGHTS),
        }
    )


def _thumb_b64(path: Path, height: int = 128) -> str:
    with Image.open(path) as im:
        im.thumbnail((height, height))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def review(embedder_kind: str, device: str, max_thumbs_per_cam: int = 2) -> None:
    """Stage 2: coherence-scored, camera-grouped identity report + decisions template.

    Args:
        embedder_kind: ``"clipreid"`` (deployed encoder — coherence scores and
            duplicate detection) or ``"none"`` (contact sheets only).
        device: Torch device for the embedder.
        max_thumbs_per_cam: Thumbnails per camera per identity in the report.
    """
    import numpy as np

    rows = _load_manifest()
    by_pid: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_pid[int(r["pid"])].append(r)

    coherence: dict[int, float] = {}
    centroids: dict[int, np.ndarray] = {}
    if embedder_kind == "clipreid":
        import cv2

        embedder = _clipreid_embedder(device)
        for pid, items in by_pid.items():
            sample = items[:: max(1, len(items) // 16)][:16]  # cap embeds per id
            crops = [cv2.imread(str(STAGING / "crops" / r["file"])) for r in sample]
            feats = embedder.embed(crops)  # L2-normalised
            centroid = feats.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            coherence[pid] = float((feats @ centroid).mean())
            centroids[pid] = centroid
        logger.info("embedded %d identities for coherence scoring", len(coherence))

    # Suspected duplicates: same-scenario identity pairs with similar centroids.
    duplicates: list[tuple[int, int, float]] = []
    if centroids:
        pids = sorted(centroids)
        for i, a in enumerate(pids):
            for b in pids[i + 1 :]:
                if a // PID_SCENARIO_BASE != b // PID_SCENARIO_BASE:
                    continue
                sim = float(centroids[a] @ centroids[b])
                if sim >= 0.85:
                    duplicates.append((a, b, sim))
        duplicates.sort(key=lambda t: -t[2])

    review_dir = STAGING / "review"
    review_dir.mkdir(exist_ok=True)
    order = sorted(by_pid, key=lambda p: coherence.get(p, 1.0))
    # Detailed (thumbnail) rows only where review effort belongs: the worst
    # coherence scores plus every duplicate suspect. Full detail for all 2k+
    # identities would be a browser-killing multi-hundred-MB page.
    detail_top = 250
    detailed = set(order[:detail_top]) | {p for a, b, _ in duplicates[:100] for p in (a, b)}
    cells: list[str] = []
    compact: list[str] = []
    for pid in order:
        items = by_pid[pid]
        by_cam: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in items:
            by_cam[r["camera"]].append(r)
        score = coherence.get(pid)
        header = (
            f"<b>pid {pid}</b> (s{pid // PID_SCENARIO_BASE}, track "
            f"{pid % PID_SCENARIO_BASE}) — {len(items)} crops / {len(by_cam)} cams"
            + (f" — coherence <b>{score:.3f}</b>" if score is not None else "")
        )
        if pid not in detailed:
            compact.append(f"<li>{header}</li>")
            continue
        thumbs = []
        for cam in sorted(by_cam, key=int):
            for r in by_cam[cam][:max_thumbs_per_cam]:
                b64 = _thumb_b64(STAGING / "crops" / r["file"])
                thumbs.append(
                    f'<span style="text-align:center;display:inline-block;margin:2px">'
                    f'<img src="data:image/jpeg;base64,{b64}"><br><small>c{cam}</small></span>'
                )
        cells.append(
            f"<div style='border-top:1px solid #999;padding:6px'>{header}<br>"
            + "".join(thumbs)
            + "</div>"
        )

    dup_html = "".join(
        f"<li>pid {a} ~ pid {b} (cosine {sim:.3f}) — consider <code>merge:{a}</code></li>"
        for a, b, sim in duplicates[:100]
    )
    (review_dir / "identity_report.html").write_text(
        "<h1>MTMDC ReID identity review</h1>"
        "<p>Sorted by embedding coherence (worst first). Check: is each row ONE person, "
        "consistent across cameras? Record actions in decisions.csv.</p>"
        f"<h2>Suspected duplicate identities (same scenario)</h2><ul>{dup_html}</ul>"
        f"<h2>Identities needing review (worst {detail_top} + duplicate suspects)</h2>"
        + "".join(cells)
        + "<h2>Remaining identities (high coherence — skim only)</h2><ul>"
        + "".join(compact)
        + "</ul>",
        encoding="utf-8",
    )

    decisions = review_dir / "decisions.csv"
    if not decisions.exists():  # never clobber a review in progress
        with open(decisions, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["pid", "coherence", "action"])
            for pid in order:
                writer.writerow([pid, f"{coherence.get(pid, float('nan')):.4f}", "keep"])
    logger.info(
        "review ready: %s (%d identities, %d suspected duplicates) — edit %s",
        review_dir / "identity_report.html",
        len(by_pid),
        len(duplicates),
        decisions,
    )


def prune(device: str) -> None:
    """Score every crop's similarity to its identity centroid (fragment filter).

    Leg-only / heavily-truncated crops are correctly-annotated partial-
    visibility boxes, so they cannot be filtered by flags (``occluded`` is set
    on 70% of crops); but they are embedding OUTLIERS within their identity.
    This embeds all crops (deployed CLIP-ReID engine), computes per-identity
    centroids, and writes ``review/crop_sims.csv`` (file, pid, sim). ``package``
    can then drop crops below ``--prune-threshold`` — pick it by eyeballing the
    montage bands from the log output.

    Args:
        device: Torch/TRT device for the embedder.
    """
    import cv2
    import numpy as np

    by_pid: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in _load_manifest():
        by_pid[int(r["pid"])].append(r)
    embedder = _clipreid_embedder(device)

    out_path = STAGING / "review" / "crop_sims.csv"
    out_path.parent.mkdir(exist_ok=True)
    sims_all: list[float] = []
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "pid", "sim"])
        for i, (pid, items) in enumerate(sorted(by_pid.items())):
            crops = [cv2.imread(str(STAGING / "crops" / r["file"])) for r in items]
            feats = embedder.embed(crops)  # L2-normalised
            centroid = feats.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            sims = feats @ centroid
            for r, s in zip(items, sims, strict=True):
                writer.writerow([r["file"], pid, f"{float(s):.4f}"])
            sims_all.extend(float(s) for s in sims)
            if i % 200 == 0:
                logger.info("pruning scores: %d/%d identities", i, len(by_pid))
    arr = np.asarray(sims_all)
    for t in (0.3, 0.35, 0.4, 0.45, 0.5):
        logger.info(
            "crops below sim %.2f: %d (%.1f%%)",
            t,
            int((arr < t).sum()),
            100 * float((arr < t).mean()),
        )
    logger.info("wrote %s (%d crops)", out_path, len(arr))


def package(out_version: str, prune_threshold: float = 0.0) -> None:
    """Stage 3: apply decisions (and optional crop pruning); write Market-1501 layout.

    Args:
        out_version: Output version id (``persondet_<id>``).
        prune_threshold: Drop crops whose identity-centroid similarity (from
            the ``prune`` stage) is below this value; ``0`` disables.
    """
    decisions_path = STAGING / "review" / "decisions.csv"
    actions: dict[int, str] = {}
    if decisions_path.exists():
        with open(decisions_path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                actions[int(r["pid"])] = r["action"].strip()
    else:
        logger.warning("no decisions.csv — packaging everything as 'keep'")

    crop_sims: dict[str, float] = {}
    sims_path = STAGING / "review" / "crop_sims.csv"
    if prune_threshold > 0:
        if not sims_path.exists():
            raise FileNotFoundError(f"--prune-threshold set but {sims_path} missing — run prune")
        with open(sims_path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                crop_sims[r["file"]] = float(r["sim"])

    out = mtmdc.DATA_ROOT / f"persondet_{out_version}"
    for sub in ("bounding_box_train", "query", "bounding_box_test"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    def resolve(pid: int) -> int | None:
        action = actions.get(pid, "keep")
        if action == "drop":
            return None
        if action.startswith("merge:"):
            return int(action.split(":", 1)[1])
        return pid

    stats = {"train": 0, "query": 0, "gallery": 0, "dropped": 0, "pruned": 0}
    train_ids: set[int] = set()
    eval_ids: set[int] = set()
    query_cam: dict[int, int] = {}
    for r in sorted(_load_manifest(), key=lambda r: (int(r["pid"]), int(r["camera"]))):
        pid = resolve(int(r["pid"]))
        if pid is None:
            stats["dropped"] += 1
            continue
        if prune_threshold > 0 and crop_sims.get(r["file"], 1.0) < prune_threshold:
            stats["pruned"] += 1  # fragment/outlier crop (e.g. legs-only)
            continue
        scenario, camera = int(r["scenario"]), int(r["camera"])
        name = f"{pid}_c{camera}_s{scenario:02d}_f{int(r['frame']):06d}.jpg"
        src = STAGING / "crops" / r["file"]
        if scenario in HELDOUT_SCENARIOS:
            eval_ids.add(pid)
            if query_cam.setdefault(pid, camera) == camera:
                dest, key = out / "query" / name, "query"
            else:
                dest, key = out / "bounding_box_test" / name, "gallery"
        else:
            train_ids.add(pid)
            dest, key = out / "bounding_box_train" / name, "train"
        mtmdc.hardlink(src, dest)
        stats[key] += 1

    (out / "README.md").write_text(
        f"""# persondet_{out_version} — MTMDC ReID crops (Market-1501 layout)

Identity-labelled person crops cut from the timeline-REALIGNED MTMDC labels
(`curation/build_reid_crops.py`; identities from the NIA JSON global person id:
pid = scenario*{PID_SCENARIO_BASE} + nia_pid). Human-reviewed via
`_reid_staging/review/` (decisions applied:
{sum(1 for a in actions.values() if a != "keep")} non-keep).

- train: {stats["train"]} crops / {len(train_ids)} identities
  (scenarios != {HELDOUT_SCENARIOS})
- query: {stats["query"]} crops, gallery: {stats["gallery"]} crops /
  {len(eval_ids)} identities (held-out scenarios {HELDOUT_SCENARIOS};
  query = first camera per identity, gallery = the rest)
- dropped in review: {stats["dropped"]} crops; pruned outlier crops
  (centroid sim < {prune_threshold}): {stats["pruned"]}

Splits are IDENTITY-disjoint by scenario (ReID protocol) — the inverse of the
detector's camera split. Same-actor reuse across scenarios becomes distinct
identities (accepted label noise). License: PIASPACE internal (NIA MTMDC).
""",
        encoding="utf-8",
    )
    logger.info("packaged persondet_%s: %s", out_version, stats)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="Cut sampled track crops into _reid_staging (CPU).")
    r = sub.add_parser("review", help="Coherence-scored HTML review + decisions template.")
    r.add_argument("--embedder", choices=["clipreid", "none"], default="clipreid")
    r.add_argument("--device", default="cuda:0", help="Device for the review embedder.")
    pr = sub.add_parser("prune", help="Score crop-to-identity-centroid similarity (GPU).")
    pr.add_argument("--device", default="cuda:0", help="Device for the embedder.")
    q = sub.add_parser("package", help="Apply decisions.csv; write Market-1501 layout.")
    q.add_argument("--out", default="reid_v1", help="Output version id.")
    q.add_argument(
        "--prune-threshold",
        type=float,
        default=0.0,
        help="Drop crops with centroid similarity below this (needs the prune stage; 0=off).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    match args.cmd:
        case "build":
            build()
        case "review":
            review(args.embedder, args.device)
        case "prune":
            prune(args.device)
        case "package":
            package(args.out, args.prune_threshold)


if __name__ == "__main__":
    main()
