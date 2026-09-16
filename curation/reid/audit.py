"""Structural audit of a Market-1501-layout ReID dataset.

Checks the things that silently break ReID training or make an eval number a lie:

* **filenames parse** — pid/camid are read from the name, so an unparseable name
  is a silently dropped crop
* **split identity-disjointness** — the ReID protocol requires train identities
  to be absent from query/gallery. A leak makes mAP meaningless (this is exactly
  what ``pia_aihub_v1`` deliberately does, and its README says so)
* **query ⊆ gallery identities** — a query whose identity has no gallery match
  can never be answered and silently drags mAP down
* **cross-camera coverage** — ReID is a cross-camera task; an identity confined
  to one camera teaches nothing and, in query/gallery, is excluded by the
  standard same-camera filter
* **crop integrity** — unreadable, empty, or degenerate images
* **identity size distribution** — singleton identities cannot form a positive
  pair for triplet loss

Run it on any Market-1501-layout directory, whoever built it:

    python -m curation.reid.audit --dataset assets/data/reid/<name>
    python -m curation.reid.audit --dataset assets/data/reid/<name> --sample-images 400
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

logger = logging.getLogger("curation.audit_reid")

SPLITS = ("bounding_box_train", "query", "bounding_box_test")
NAME_RE = re.compile(r"^(-?\d+)_c(\d+)")
# Below this a crop is upscaling noise at the 256x128 input, not appearance signal.
MIN_USEFUL_H = 64
MIN_USEFUL_W = 24


def scan(split_dir: Path) -> tuple[list[tuple[Path, int, int]], list[Path]]:
    """Parse ``<pid>_c<camid>`` out of every image name in one split.

    Returns:
        ``(parsed, unparseable)`` where parsed items are ``(path, pid, camid)``.
    """
    parsed: list[tuple[Path, int, int]] = []
    bad: list[Path] = []
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = NAME_RE.match(p.name)
        if m is None:
            bad.append(p)
        else:
            parsed.append((p, int(m.group(1)), int(m.group(2))))
    return parsed, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--sample-images",
        type=int,
        default=600,
        help="how many crops to actually decode for integrity checks",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = args.dataset
    print(f"\n{'=' * 72}\n{root.name}\n{'=' * 72}")

    data: dict[str, list[tuple[Path, int, int]]] = {}
    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            print(f"  MISSING SPLIT: {split}")
            continue
        parsed, bad = scan(d)
        data[split] = parsed
        pids = {pid for _, pid, _ in parsed}
        cams = {cam for _, _, cam in parsed}
        junk = sum(1 for _, pid, _ in parsed if pid == -1)
        print(f"\n  {split}")
        print(f"    crops        {len(parsed)}")
        print(
            f"    identities   {len(pids - {-1})}"
            + (f"  (+ junk pid -1: {junk} crops)" if junk else "")
        )
        print(f"    cameras      {sorted(cams)}")
        if bad:
            print(f"    UNPARSEABLE NAMES: {len(bad)}  e.g. {[p.name for p in bad[:3]]}")

    # --- split disjointness (the ReID protocol requirement) -------------------
    print("\n  --- split integrity ---")
    train_pids = {pid for _, pid, _ in data.get("bounding_box_train", []) if pid != -1}
    query_pids = {pid for _, pid, _ in data.get("query", []) if pid != -1}
    gal_pids = {pid for _, pid, _ in data.get("bounding_box_test", []) if pid != -1}

    leak = train_pids & (query_pids | gal_pids)
    if leak:
        print(f"    NOT identity-disjoint: {len(leak)} identities appear in train AND eval")
        print(f"      e.g. {sorted(leak)[:8]}")
    else:
        print("    identity-disjoint: train shares no identity with query/gallery  [OK]")

    missing = query_pids - gal_pids
    print(
        f"    query identities absent from gallery: {len(missing)}"
        + (f"  e.g. {sorted(missing)[:6]}" if missing else "  [OK]")
    )

    # --- cross-camera coverage ----------------------------------------------
    print("\n  --- cross-camera coverage ---")
    for label, split in (("train", "bounding_box_train"), ("gallery", "bounding_box_test")):
        rows = [r for r in data.get(split, []) if r[1] != -1]
        cams_per: dict[int, set[int]] = defaultdict(set)
        for _, pid, cam in rows:
            cams_per[pid].add(cam)
        single = sum(1 for v in cams_per.values() if len(v) < 2)
        if cams_per:
            print(
                f"    {label:8s} identities on a single camera: {single}/{len(cams_per)}"
                f"  ({100 * single / len(cams_per):.1f}%)"
            )

    # --- identity size distribution -----------------------------------------
    print("\n  --- crops per identity (train) ---")
    counts = Counter(pid for _, pid, _ in data.get("bounding_box_train", []) if pid != -1)
    if counts:
        vals = sorted(counts.values())
        singles = sum(1 for v in vals if v < 2)
        print(f"    min {vals[0]}  median {vals[len(vals) // 2]}  max {vals[-1]}")
        print(f"    identities with <2 crops (no positive pair possible): {singles}")
        print(
            f"    identities with <4 crops (below num_instances=4):     "
            f"{sum(1 for v in vals if v < 4)}"
        )

    # --- crop integrity (sampled) -------------------------------------------
    print(f"\n  --- crop integrity (sampling {args.sample_images}) ---")
    everything = [p for rows in data.values() for p, _, _ in rows]
    random.Random(args.seed).shuffle(everything)
    sample = everything[: args.sample_images]
    unreadable: list[Path] = []
    tiny: list[tuple[str, int, int]] = []
    sizes: list[tuple[int, int]] = []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            unreadable.append(p)
            continue
        h, w = img.shape[:2]
        sizes.append((w, h))
        if h < MIN_USEFUL_H or w < MIN_USEFUL_W:
            tiny.append((p.name, w, h))
    print(
        f"    unreadable: {len(unreadable)}"
        + (f"  e.g. {[p.name for p in unreadable[:3]]}" if unreadable else "  [OK]")
    )
    print(
        f"    below {MIN_USEFUL_W}x{MIN_USEFUL_H}px: {len(tiny)}"
        + (f"  e.g. {tiny[:3]}" if tiny else "  [OK]")
    )
    if sizes:
        hs = sorted(h for _, h in sizes)
        ws = sorted(w for w, _ in sizes)
        print(f"    height  min {hs[0]}  median {hs[len(hs) // 2]}  max {hs[-1]}")
        print(f"    width   min {ws[0]}  median {ws[len(ws) // 2]}  max {ws[-1]}")


if __name__ == "__main__":
    main()
