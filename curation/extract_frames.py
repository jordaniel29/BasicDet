"""Stage 1 — extract subsampled frames from the MTMDC videos into a shared pool.

For every camera, keep every ``SUBSAMPLE_EVERY``-th annotated frame (the temporal
subsampling that *is* the dedup for fixed cameras — curation.md sec.4/5), decode
it from the ``.avi`` and write it as a JPEG into::

    FRAME_POOL/<scenario>/<camera>/<source>_sNN_cNN_FFFFFF.jpg

The pool is built ONCE and hardlinked into each curated version (v4.1, v4.2) by
``build_version.py`` — both versions draw from the same frames, only their split
assignment differs, so this avoids decoding or storing the frames twice.

A ``manifest.csv`` (one row per extracted frame: scenario, camera, frame, W, H,
n_boxes) is written alongside the pool and is the authoritative list of extracted
frames that ``build_version.py`` consumes.

The frame index equals the 0-based video decode index (verified by overlaying
VATIC boxes on decoded frames). We decode sequentially with ``grab()`` and only
``retrieve()`` the kept frames — reliable on AVI and far cheaper than seeking.

Usage (run from the repo root):
    python -m curation.extract_frames                 # all 352 cameras
    python -m curation.extract_frames --scenario 1    # one scenario (smoke test)
    python -m curation.extract_frames --jobs 12 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from curation import mtmdc

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

logger = logging.getLogger("mtmdc.extract")

MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = ("scenario", "camera", "frame", "width", "height", "n_boxes", "rel_path")


def _kept_frames(scenario: int, camera: int, vatic: dict[int, list[mtmdc.Box]]) -> list[int]:
    """Every ``SUBSAMPLE_EVERY``-th annotated frame to extract.

    Empty (no-box) frames are dropped unless ``KEEP_NEGATIVES`` — MTMDC's
    zero-label frames are contaminated with unlabeled bystanders (see the flag's
    docstring in ``mtmdc.py``).
    """
    universe = mtmdc.annotated_frame_indices(scenario, camera)
    grid = [f for f in universe if f % mtmdc.SUBSAMPLE_EVERY == 0]
    if mtmdc.KEEP_NEGATIVES:
        return grid
    return [f for f in grid if vatic.get(f)]


def _pool_jpg(scenario: int, camera: int, frame: int) -> Path:
    stem = mtmdc.frame_stem(scenario, camera, frame)
    return (
        mtmdc.FRAME_POOL / mtmdc.scenario_name(scenario) / mtmdc.camera_name(camera) / f"{stem}.jpg"
    )


def extract_camera(scenario: int, camera: int, skip_existing: bool) -> dict:
    """Decode and write every kept frame for one camera; return manifest rows + stats.

    Args:
        scenario: Scenario number (e.g. ``1``).
        camera: Camera number (1-16).
        skip_existing: If True, do not re-decode/overwrite frames whose JPEG
            already exists (resumable re-runs); their manifest rows are still
            emitted.

    Returns:
        ``{"rows": [...], "written": int, "skipped": int, "missing": int}``.

    Raises:
        FileNotFoundError: If the camera video is missing.
        RuntimeError: If the video cannot be opened.
    """
    video = mtmdc.camera_video(scenario, camera)
    if not video.is_file():
        raise FileNotFoundError(f"missing video: {video}")

    vatic = mtmdc.parse_vatic(mtmdc.camera_vatic(scenario, camera))
    kept = _kept_frames(scenario, camera, vatic)
    out_dir = _pool_jpg(scenario, camera, 0).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # n_boxes is computed from the cleaned VATIC boxes; W,H are filled in from the
    # first decoded (or cached) frame, since the box geometry depends on them.
    rows: list[dict] = []
    n_boxes_by_frame = {f: vatic.get(f, []) for f in kept}

    # Fast path: everything already on disk -> read W,H once and emit rows.
    targets = {f: _pool_jpg(scenario, camera, f) for f in kept}
    if skip_existing and kept and all(p.exists() for p in targets.values()):
        sample = cv2.imread(str(targets[kept[0]]))
        if sample is None:
            raise RuntimeError(f"cannot read cached frame: {targets[kept[0]]}")
        h, w = sample.shape[:2]
        for f in kept:
            n = len(mtmdc.clean_boxes(n_boxes_by_frame[f], w, h))
            rows.append(_row(scenario, camera, f, w, h, n, targets[f]))
        return {"rows": rows, "written": 0, "skipped": len(kept), "missing": 0}

    kept_set = set(kept)
    max_keep = max(kept) if kept else -1
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    written = skipped = missing = dropped = 0
    width = height = 0
    idx = 0
    try:
        while idx <= max_keep:
            if not cap.grab():  # advance decoder; cheap, no color copy
                break
            if idx in kept_set:
                ok, frame = cap.retrieve()
                if not ok:
                    missing += 1
                    idx += 1
                    continue
                height, width = frame.shape[:2]
                n = len(mtmdc.clean_boxes(n_boxes_by_frame[idx], width, height))
                if n == 0 and not mtmdc.KEEP_NEGATIVES:
                    # All boxes were degenerate (<=1px) -> treat as empty, drop it.
                    dropped += 1
                    idx += 1
                    continue
                dst = targets[idx]
                if skip_existing and dst.exists():
                    skipped += 1
                else:
                    cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, mtmdc.JPEG_QUALITY])
                    written += 1
                rows.append(_row(scenario, camera, idx, width, height, n, dst))
            idx += 1
    finally:
        cap.release()

    expected = len(kept)
    got = written + skipped + dropped
    if got < expected:
        missing += expected - got
        logger.warning(
            "%s/%s: decoded only %d/%d kept frames (video shorter than annotations?)",
            mtmdc.scenario_name(scenario),
            mtmdc.camera_name(camera),
            got,
            expected,
        )
    return {"rows": rows, "written": written, "skipped": skipped, "missing": missing}


def _row(scenario: int, camera: int, frame: int, w: int, h: int, n: int, jpg: Path) -> dict:
    return {
        "scenario": scenario,
        "camera": camera,
        "frame": frame,
        "width": w,
        "height": h,
        "n_boxes": n,
        "rel_path": str(jpg.relative_to(mtmdc.FRAME_POOL)),
    }


def _task(args: tuple[int, int, bool]) -> dict:
    scenario, camera, skip = args
    res = extract_camera(scenario, camera, skip)
    res["scenario"] = scenario
    res["camera"] = camera
    return res


def discover_tasks(scenarios: list[int], cameras: list[int]) -> list[tuple[int, int]]:
    """All (scenario, camera) pairs whose video exists on disk."""
    pairs: list[tuple[int, int]] = []
    for s in scenarios:
        for c in cameras:
            if mtmdc.camera_video(s, c).is_file():
                pairs.append((s, c))
            else:
                logger.warning(
                    "skipping missing video %s/%s", mtmdc.scenario_name(s), mtmdc.camera_name(c)
                )
    return pairs


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    scenarios = args.scenario or list(mtmdc.ALL_SCENARIOS)
    cameras = args.camera or list(mtmdc.CAMERAS)
    pairs = discover_tasks(scenarios, cameras)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("no cameras to process")

    mtmdc.FRAME_POOL.mkdir(parents=True, exist_ok=True)
    logger.info(
        "extracting every %dth frame from %d cameras -> %s (jobs=%d, skip_existing=%s)",
        mtmdc.SUBSAMPLE_EVERY,
        len(pairs),
        mtmdc.FRAME_POOL,
        args.jobs,
        args.skip_existing,
    )

    tasks = [(s, c, args.skip_existing) for (s, c) in pairs]
    all_rows: list[dict] = []
    tot = {"written": 0, "skipped": 0, "missing": 0}
    bar = tqdm(total=len(tasks), unit="cam") if (tqdm and not args.no_progress) else None

    def _accumulate(res: dict) -> None:
        all_rows.extend(res["rows"])
        for k in tot:
            tot[k] += res[k]
        if bar:
            bar.update(1)
            bar.set_postfix(frames=tot["written"] + tot["skipped"], refresh=False)

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_task, t) for t in tasks]
            for fut in as_completed(futs):
                _accumulate(fut.result())
    else:
        for t in tasks:
            _accumulate(_task(t))
    if bar:
        bar.close()

    all_rows.sort(key=lambda r: (r["scenario"], r["camera"], r["frame"]))
    manifest = mtmdc.FRAME_POOL / MANIFEST_NAME
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(
        "done: %d frames in pool (%d written, %d skipped, %d missing); manifest -> %s",
        len(all_rows),
        tot["written"],
        tot["skipped"],
        tot["missing"],
        manifest,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--scenario",
        type=int,
        nargs="*",
        default=None,
        help="Limit to these scenario numbers (default: all 22).",
    )
    p.add_argument(
        "--camera",
        type=int,
        nargs="*",
        default=None,
        help="Limit to these camera numbers (default: 1-16).",
    )
    p.add_argument("--limit", type=int, default=0, help="Process only the first N cameras (debug).")
    p.add_argument("--jobs", type=int, default=10, help="Parallel worker processes.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-decode frames whose JPEG already exists (resumable).",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    return p.parse_args()


if __name__ == "__main__":
    main()
