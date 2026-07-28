"""Convert the ``atustc9`` per-identity crop dump into the Market-1501 layout.

``atustc9`` ships as one folder per (person, day/session, session-camera-group)::

    atustc9/
    └── p001-d03-c12/
        └── cam09-f0-0075.jpg   # <camera>-<tracklet>-<frame-in-camera>

The identity is the ``pNNN`` folder prefix. The REAL camera is the FILENAME
prefix (``cam01``..``cam16``, 16 cameras) — NOT the folder's ``cNN`` (that
tracks the day/session; a single person-day folder mixes many cameras). ``fN``
is a per-camera tracklet index (irrelevant to identity, kept only for filename
uniqueness).

The ``basicdet`` ReID trainers (ft_net and the official CLIP-ReID) consume the
Market-1501 layout (``bounding_box_train/``, ``query/``, ``bounding_box_test/``;
filenames ``<pid>_c<camid>_...jpg``), so this script rewrites ``atustc9`` into
that layout via **hardlinks** (zero extra disk — curation.md sec.1):

  * **Identity-disjoint split** (standard ReID protocol): ``--num-eval-ids``
    identities that appear in >=2 cameras are held out for query/gallery
    (query = one crop per (identity, camera); gallery = the rest, so every
    query has cross-camera positives). All remaining identities train.
  * **Per-camera cap**: at most ``--cap-per-camera`` evenly-spaced crops per
    (identity, camera). ``atustc9`` is ~50x imbalanced (191..9735 crops/id);
    the near-duplicate frames add little, so capping flattens the imbalance and
    cuts two-stage training time with negligible accuracy loss.

Usage (from the repo root, conda env ``persondet``)::

    python -m curation.build_atustc9_reid          # -> assets/data/persondet_reid_atustc9
    python -m curation.build_atustc9_reid --cap-per-camera 40 --num-eval-ids 50
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from curation.mtmdc import hardlink

logger = logging.getLogger("atustc9.reid")

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = _REPO_ROOT / "assets" / "data" / "atustc9"
DEFAULT_OUT = _REPO_ROOT / "assets" / "data" / "persondet_reid_atustc9"

# Folder ``p001-d03-c12`` and file ``cam09-f0-0075.jpg``.
_DIR_RE = re.compile(r"^p(\d+)-d(\d+)-c(\d+)$")
_FILE_RE = re.compile(r"^cam(\d+)-f(\d+)-(\d+)$")

CAP_PER_CAMERA = 40  # evenly-spaced crops kept per (identity, camera)
NUM_EVAL_IDS = 50  # identities (>=2 cameras) held out for query/gallery
MIN_EVAL_CAMERAS = 2  # an eval identity needs >=2 cameras for cross-camera eval


def _evenly_spaced(items: list, cap: int) -> list:
    """Keep at most ``cap`` items, evenly spaced across ``items`` (order kept)."""
    if cap <= 0 or len(items) <= cap:
        return items
    return [items[(i * len(items)) // cap] for i in range(cap)]


def scan_crops(src: Path) -> dict[tuple[int, int], list[Path]]:
    """Index every crop by ``(pid, camid)``.

    Args:
        src: The ``atustc9`` root (folders ``pNNN-dNN-cNN``).

    Returns:
        Mapping ``(pid, camid) -> sorted list of image paths``. ``pid`` is the
        folder's person number; ``camid`` is the filename camera (1..16).

    Raises:
        FileNotFoundError: If ``src`` is not a directory.
        ValueError: If no crops are found (wrong path / empty dump).
    """
    if not src.is_dir():
        raise FileNotFoundError(f"atustc9 source not found: {src}")

    by_key: dict[tuple[int, int], list[Path]] = defaultdict(list)
    n_skipped = 0
    for person_dir in sorted(src.iterdir()):
        dir_m = _DIR_RE.match(person_dir.name)
        if not (person_dir.is_dir() and dir_m):
            continue
        pid = int(dir_m.group(1))
        for img in person_dir.glob("*.jpg"):
            file_m = _FILE_RE.match(img.stem)
            if file_m is None:
                n_skipped += 1
                continue
            camid = int(file_m.group(1))
            by_key[(pid, camid)].append(img)

    if not by_key:
        raise ValueError(f"no crops matched the atustc9 naming under {src}")
    for paths in by_key.values():
        paths.sort()  # deterministic: sort by (day, tracklet, frame) via filename
    if n_skipped:
        logger.warning("skipped %d files not matching cam<NN>-f<N>-<frame>.jpg", n_skipped)
    return by_key


def choose_eval_ids(by_key: dict[tuple[int, int], list[Path]], num_eval_ids: int) -> set[int]:
    """Pick evenly-spaced identities (>=2 cameras) to hold out for query/gallery.

    Args:
        by_key: The ``(pid, camid) -> paths`` index from :func:`scan_crops`.
        num_eval_ids: How many identities to reserve for evaluation.

    Returns:
        The held-out identity ids. Empty if ``num_eval_ids == 0``.
    """
    cams_per_pid: dict[int, set[int]] = defaultdict(set)
    for pid, camid in by_key:
        cams_per_pid[pid].add(camid)
    eligible = sorted(pid for pid, cams in cams_per_pid.items() if len(cams) >= MIN_EVAL_CAMERAS)
    if num_eval_ids <= 0 or not eligible:
        return set()
    if num_eval_ids >= len(eligible):
        logger.warning(
            "requested %d eval ids but only %d have >=%d cameras — holding out all",
            num_eval_ids,
            len(eligible),
            MIN_EVAL_CAMERAS,
        )
        return set(eligible)
    # Evenly spaced across the sorted eligible ids -> representative holdout.
    return {eligible[(i * len(eligible)) // num_eval_ids] for i in range(num_eval_ids)}


def _dest_name(pid: int, camid: int, src: Path) -> str:
    """Market-1501 filename ``<pid>_c<camid>_<source-tag>.jpg`` (globally unique).

    The source folder + original stem is unique, so it disambiguates crops that
    share a (pid, camid, frame) across different day/session folders.
    """
    return f"{pid}_c{camid}_{src.parent.name}-{src.stem}.jpg"


def build(src: Path, out: Path, cap_per_camera: int, num_eval_ids: int) -> None:
    """Write the Market-1501 layout from ``atustc9`` via hardlinks.

    Args:
        src: The ``atustc9`` root.
        out: Output dataset dir (``bounding_box_train/`` etc. are (re)created).
        cap_per_camera: Max evenly-spaced crops per (identity, camera).
        num_eval_ids: Identities held out for query/gallery.
    """
    by_key = scan_crops(src)
    eval_ids = choose_eval_ids(by_key, num_eval_ids)

    for sub in ("bounding_box_train", "query", "bounding_box_test"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "query": 0, "gallery": 0}
    train_ids: set[int] = set()
    # Per (pid, camid): the first kept crop is the query probe, the rest gallery.
    for (pid, camid), paths in sorted(by_key.items()):
        kept = _evenly_spaced(paths, cap_per_camera)
        if pid in eval_ids:
            for i, p in enumerate(kept):
                split = "query" if i == 0 else "bounding_box_test"
                hardlink(p, out / split / _dest_name(pid, camid, p))
            stats["query"] += 1
            stats["gallery"] += len(kept) - 1
        else:
            for p in kept:
                hardlink(p, out / "bounding_box_train" / _dest_name(pid, camid, p))
            stats["train"] += len(kept)
            train_ids.add(pid)

    (out / "README.md").write_text(
        f"""# persondet_reid_atustc9 — atustc9 ReID crops (Market-1501 layout)

Built by `curation/build_atustc9_reid.py` from `assets/data/atustc9` (hardlinks).
Identity = `pNNN` folder; camera = filename `cam<NN>` (1..16, the real camera —
the folder `cNN` is the day/session, not a camera).

- train:   {stats["train"]} crops / {len(train_ids)} identities
- query:   {stats["query"]} crops (one per held-out identity+camera)
- gallery: {stats["gallery"]} crops / {len(eval_ids)} identities
  (held-out identities with >={MIN_EVAL_CAMERAS} cameras; cross-camera protocol)

Splits are IDENTITY-disjoint (standard ReID protocol). At most
{cap_per_camera} evenly-spaced crops per (identity, camera). License: internal.
""",
        encoding="utf-8",
    )
    logger.info(
        "packaged %s: %s (%d train ids, %d eval ids)",
        out.name,
        stats,
        len(train_ids),
        len(eval_ids),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="atustc9 source root.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output dataset dir.")
    p.add_argument(
        "--cap-per-camera",
        type=int,
        default=CAP_PER_CAMERA,
        help="Max evenly-spaced crops per (identity, camera).",
    )
    p.add_argument(
        "--num-eval-ids",
        type=int,
        default=NUM_EVAL_IDS,
        help="Identities (>=2 cameras) held out for query/gallery (0 = train on all).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    build(args.src, args.out, args.cap_per_camera, args.num_eval_ids)


if __name__ == "__main__":
    main()
