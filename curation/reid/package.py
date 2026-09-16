"""Write a Market-1501 layout from staged crops, and refuse to write a leaky one.

The layout is three flat directories — ``bounding_box_train``, ``query``,
``bounding_box_test`` — whose crop names carry the identity and camera. Every
ReID loader in this repo and in TRACE reads them that way.

Two rules are enforced in code rather than documented, because breaking either
produces a number that looks fine and means nothing:

* **train and eval must not share an identity.** A shared identity turns the
  reported mAP into a measure of memorisation. Two datasets here shipped with
  that defect before it was checked at build time.
* **every query identity must have gallery crops on another camera.** The Market
  protocol drops same-camera gallery entries per query, so a query whose
  identity appears on one camera only can never be answered and silently drags
  the average down.

Crops are **hardlinked** from the staging area, so a packaged set costs no extra
disk and the staging area stays the single source of truth.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("curation.reid.package")

SPLITS = ("bounding_box_train", "query", "bounding_box_test")
TRAIN, QUERY, GALLERY = SPLITS


def load_decisions(path: Path) -> dict[int, str]:
    """Read a human review file mapping identity to action.

    The vocabulary is shared by every curation recipe here: ``keep`` (default),
    ``drop``, or ``merge:<pid>`` to fold one identity into another.

    Args:
        path: ``pid,action[,note]`` CSV. A missing file means "keep everything",
            which is a normal state before anyone has reviewed.

    Returns:
        ``{pid: action}``; empty if the file does not exist.
    """
    if not path.is_file():
        logger.warning("no decisions file at %s — treating every identity as 'keep'", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        return {int(r["pid"]): r["action"].strip() for r in csv.DictReader(fh)}


def resolve_pid(pid: int, actions: dict[int, str]) -> int | None:
    """Apply a review decision to one identity.

    Returns:
        The identity to write it under, or ``None`` if it was dropped.

    Raises:
        ValueError: On an unrecognised action, so a typo in a hand-edited
            decisions file cannot silently become "keep".
    """
    action = actions.get(pid, "keep")
    if action == "keep":
        return pid
    if action == "drop":
        return None
    if action.startswith("merge:"):
        return int(action.split(":", 1)[1])
    raise ValueError(f"pid {pid}: unknown review action {action!r} (keep|drop|merge:<pid>)")


def assign_split(pid: int, camera: int, heldout: tuple[int, ...], query_camera: int) -> str:
    """Route one crop to a split.

    Held-out identities form the evaluation set, divided by camera: one camera
    supplies the queries and the rest the gallery. Everything else trains.
    """
    if pid not in heldout:
        return TRAIN
    return QUERY if camera == query_camera else GALLERY


def write_market_layout(
    dataset_dir: Path,
    crops: list[tuple[Path, int, int, str, str]],
    strict: bool = True,
) -> dict[str, int]:
    """Hardlink staged crops into ``dataset_dir`` as a verified Market-1501 set.

    The split of each crop is decided by the caller, not here: identity holdout
    (:func:`assign_split`) is the common rule, but sources have also been split by
    scenario and by recording session. That is exactly why the checks below matter
    — a hand-rolled split rule is where leaks come from.

    Args:
        dataset_dir: Destination root. Its three split directories are emptied
            first, so re-running is idempotent.
        crops: ``(source_path, pid, camera, packaged_name, split)`` per crop, with
            review decisions already applied. ``split`` must be one of
            :data:`SPLITS`.
        strict: Run the two integrity checks. Only turn this off to inspect a
            split you already know is leaky.

    Returns:
        Crop counts keyed by split name.

    Raises:
        ValueError: On an unknown split name.
        RuntimeError: If train and eval share an identity, or a query identity has
            no gallery crops on a different camera.
    """
    for split in SPLITS:
        d = dataset_dir / split
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.iterdir():
            stale.unlink()

    counts: dict[str, int] = defaultdict(int)
    split_pids: dict[str, set[int]] = defaultdict(set)
    cameras: dict[tuple[str, int], set[int]] = defaultdict(set)

    for source, pid, camera, name, split in crops:
        if split not in SPLITS:
            raise ValueError(f"{name}: unknown split {split!r}, expected one of {SPLITS}")
        (dataset_dir / split / name).hardlink_to(source)
        counts[split] += 1
        split_pids[split].add(pid)
        cameras[(split, pid)].add(camera)

    if strict:
        verify_splits(split_pids, cameras)

    logger.info(
        "packaged -> %s | train %d ids / %d crops | eval %d ids / %d query / %d gallery",
        dataset_dir,
        len(split_pids[TRAIN]),
        counts[TRAIN],
        len(split_pids[QUERY] | split_pids[GALLERY]),
        counts[QUERY],
        counts[GALLERY],
    )
    return dict(counts)


def verify_splits(
    split_pids: dict[str, set[int]],
    cameras: dict[tuple[str, int], set[int]],
) -> None:
    """Raise unless the split is one an evaluation number can be trusted from.

    Args:
        split_pids: Identities present in each split.
        cameras: Cameras each ``(split, pid)`` appears on.

    Raises:
        RuntimeError: On a train/eval identity overlap, or a query identity with
            no gallery crop on another camera.
    """
    leak = split_pids[TRAIN] & (split_pids[QUERY] | split_pids[GALLERY])
    if leak:
        raise RuntimeError(
            f"train and eval share {len(leak)} identities, e.g. {sorted(leak)[:8]} — the mAP "
            "this would report is memorisation, not skill. Check the holdout rule and any "
            "merge decisions."
        )
    unanswerable = sorted(
        pid for pid in split_pids[QUERY] if not (cameras[(GALLERY, pid)] - cameras[(QUERY, pid)])
    )
    if unanswerable:
        raise RuntimeError(
            f"query identities with no cross-camera gallery match: {unanswerable[:8]} — the "
            "Market protocol filters same-camera gallery entries, so these can never be "
            "answered and silently drag the average down."
        )
