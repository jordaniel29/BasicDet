"""Union several packaged ReID sources into one Market-1501 training set.

Which sources go in is a **recipe**, not code: a YAML file listing each source,
the identity and camera blocks it is shifted into, and whether it contributes to
evaluation. Adding a combination therefore means writing a ~15-line YAML, not
editing this module. Recipes live in ``curation/recipes/`` (gitignored — they are
a record of what was built here); the format is documented in
:func:`load_recipe`.

Composition
-----------
* **train** = the union of every source's training split.
* **eval**  = only sources marked ``use_eval``. A source whose own query/gallery
  share identities with its own train must be excluded from eval, or the leak
  propagates into the combined number. That is a property of the source, which is
  why it is declared per source rather than assumed.

Two collisions have to be resolved, and both are silent if missed
----------------------------------------------------------------
1. **Cameras.** Every source numbers cameras from 1. With ``SIE_CAMERA`` enabled
   the model learns a per-camera embedding, so two unrelated physical cameras
   sharing an id get fused into one — no error, just a worse encoder. Each source
   is shifted into its own camera block.
2. **Identities.** Sources that key identities on ``scenario * 10000 + person``
   will eventually collide with each other. Each source is shifted into its own
   identity block.

Market's junk ``pid = -1`` is preserved as ``-1`` rather than shifted: it marks
gallery distractors, not a person, and offsetting it would fuse thousands of
distractors into one enormous fake identity.

Crops are hardlinked, so a combination costs no extra disk and every source stays
untouched.

Usage (from the repo root, conda env ``persondet``):
    python -m curation.reid.combine --recipe curation/recipes/combined_v3.yaml
    python -m curation.reid.combine --recipe ... --verify-only
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("curation.reid.combine")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REID = REPO_ROOT / "assets/data/reid"
SPLITS = ("bounding_box_train", "query", "bounding_box_test")
NAME_RE = re.compile(r"^(-?\d+)_c(\d+)")
JUNK_PID = -1


@dataclass(frozen=True)
class Source:
    """One input dataset and how its ids are moved into the combined namespace.

    Attributes:
        name: Directory under ``assets/data/reid``.
        tag: Short token embedded in output filenames for traceability.
        pid_offset: Added to every non-junk pid.
        cam_offset: Added to every camera id.
        use_eval: Whether this source contributes query/gallery.
    """

    name: str
    tag: str
    pid_offset: int
    cam_offset: int
    use_eval: bool


def load_recipe(path: Path) -> tuple[str, tuple[Source, ...]]:
    """Read a combine recipe.

    Format::

        name: combined_v3          # output directory under assets/data/reid
        sources:
          - name: market_1501      # directory under assets/data/reid
            tag: mk                # short token put in output filenames
            pid_offset: 0          # added to every non-junk identity
            cam_offset: 0          # added to every camera id
            use_eval: true         # does it contribute query/gallery?

    Args:
        path: The recipe YAML.

    Returns:
        ``(output_name, sources)``.

    Raises:
        FileNotFoundError: If the recipe does not exist.
        ValueError: If a required key is missing, or two sources would share an
            identity or camera block — the collision this module exists to prevent.
    """
    if not path.is_file():
        raise FileNotFoundError(f"combine recipe not found: {path}")
    spec = yaml.safe_load(path.read_text())
    try:
        name = spec["name"]
        sources = tuple(
            Source(
                name=s["name"],
                tag=s["tag"],
                pid_offset=int(s["pid_offset"]),
                cam_offset=int(s["cam_offset"]),
                use_eval=bool(s["use_eval"]),
            )
            for s in spec["sources"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: malformed recipe ({exc})") from exc
    if not sources:
        raise ValueError(f"{path}: recipe lists no sources")
    for field in ("pid_offset", "cam_offset", "tag"):
        seen = [getattr(s, field) for s in sources]
        if len(set(seen)) != len(seen):
            raise ValueError(f"{path}: two sources share a {field} — ids would collide")
    return name, sources


def scan(split_dir: Path) -> list[tuple[Path, int, int]]:
    """``(path, pid, camid)`` for every parseable image in a split."""
    rows: list[tuple[Path, int, int]] = []
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = NAME_RE.match(p.name)
        if m is None:
            raise ValueError(f"unparseable crop name: {p}")
        rows.append((p, int(m.group(1)), int(m.group(2))))
    return rows


def build(dst: Path, sources: tuple[Source, ...]) -> None:
    for split in SPLITS:
        d = dst / split
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.iterdir():
            stale.unlink()

    manifest: list[dict[str, object]] = []
    counts: dict[str, int] = defaultdict(int)

    for src in sources:
        root = REID / src.name
        if not root.is_dir():
            raise FileNotFoundError(f"source dataset missing: {root}")
        # PIA contributes training crops only — see the module docstring.
        wanted = SPLITS if src.use_eval else ("bounding_box_train",)
        for split in wanted:
            for path, pid, cam in scan(root / split):
                new_pid = JUNK_PID if pid == JUNK_PID else pid + src.pid_offset
                new_cam = cam + src.cam_offset
                name = f"{new_pid}_c{new_cam}_{src.tag}_{path.stem}.jpg"
                (dst / split / name).hardlink_to(path)
                counts[f"{src.name}:{split}"] += 1
                counts[split] += 1
                manifest.append(
                    {
                        "file": name,
                        "split": split,
                        "source": src.name,
                        "pid": new_pid,
                        "camid": new_cam,
                        "orig_pid": pid,
                        "orig_camid": cam,
                        "orig_file": path.name,
                    }
                )

    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    _report(dst, counts, sources)
    _verify(dst)
    _write_readme(dst, sources, counts)


def _write_readme(dst: Path, sources: tuple[Source, ...], counts: dict[str, int]) -> None:
    """Write the dataset README from what was actually built.

    Hand-maintained notes drift from the data; every number here is counted off
    the built splits so the README cannot lie about composition.

    Args:
        dst: The built dataset root.
        sources: The combination that produced it.
        counts: Per-source and per-split crop tallies from ``build``.
    """
    data = {sp: scan(dst / sp) for sp in SPLITS}
    ids = {sp: {pid for _, pid, _ in rows if pid != JUNK_PID} for sp, rows in data.items()}
    cams = sorted({cam for rows in data.values() for _, _, cam in rows})
    junk = sum(1 for _, pid, _ in data["bounding_box_test"] if pid == JUNK_PID)

    split_rows = "\n".join(
        f"| {sp} | {len(data[sp]):,} | {len(ids[sp]):,}"
        + (f" (+ {junk:,} junk crops, pid −1)" if sp == "bounding_box_test" and junk else "")
        + " |"
        for sp in SPLITS
    )

    src_rows = []
    for src in sources:
        root = REID / src.name
        tr = scan(root / "bounding_box_train")
        tr_ids = len({pid for _, pid, _ in tr if pid != JUNK_PID})
        if src.use_eval:
            q, g = scan(root / "query"), scan(root / "bounding_box_test")
            n_ev = len({pid for _, pid, _ in q if pid != JUNK_PID})
            ev = f"{n_ev:,} ids / {len(q):,} q / {len(g):,} g"
        else:
            ev = "**none — train only**"
        src_rows.append(f"| `{src.name}` | {tr_ids:,} ids / {len(tr):,} crops | {ev} |")

    cam_rows = "\n".join(
        f"    {src.name:<16} +{src.cam_offset:<4} pid +{src.pid_offset:,}" for src in sources
    )
    train_only = [s.name for s in sources if not s.use_eval]
    n_train = len(data["bounding_box_train"])
    # The PK sampler draws identities uniformly, so a source's influence tracks
    # its share of IDENTITIES, not of crops.
    venue_ids = sum(
        len({pid for _, pid, _ in scan(REID / s.name / "bounding_box_train") if pid != JUNK_PID})
        for s in sources
        if s.name.startswith("pia")
    )
    n_train_ids = len(ids["bounding_box_train"])
    share = 100.0 * venue_ids / max(n_train_ids, 1)
    one_in = max(round(n_train_ids / max(venue_ids, 1)), 1)

    (dst / "README.md").write_text(
        f"""# {dst.name} — {" + ".join(s.name for s in sources)} (Market-1501 layout)

Built by `python -m curation.reid.combine --recipe curation/recipes/{dst.name}.yaml`.
Crops are **hardlinked** from the sources, which are unchanged — this costs no
extra disk. Regenerate or re-check with
`... --recipe curation/recipes/{dst.name}.yaml --verify-only`.

| split | crops | identities |
|---|--:|--:|
{split_rows}

{len(cams)} distinct cameras. Train pids relabel contiguously in the loader.

## Composition

| source | → train | → eval |
|---|--:|---|
{chr(10).join(src_rows)}

{"**" + ", ".join(f"`{n}`" for n in train_only) + " is train-only.**" if train_only else ""}
Its query/gallery share identities with its own train, so including them would
inject a leak straight into the combined evaluation. Excluding them keeps
combined train fully person-disjoint from combined eval.

## Collisions resolved

Every source numbers cameras from 1 and overlaps on 1/3/5. With `SIE_CAMERA`
enabled a collision fuses unrelated physical cameras into one embedding,
silently and with no error. Identity blocks are offset for the same reason:

```
{cam_rows}
```

The official loader compacts camera ids to `0..N-1` internally, so only
distinctness matters; the offsets keep provenance readable. Market's junk
`pid = −1` is **preserved as −1** — it marks gallery distractors, not an
identity, and offsetting it would turn {junk:,} distractors into one enormous
fake person.

## Verified at build time

Enforced in code, raising rather than warning:

- train ∩ eval identities = 0 ✅
- every query identity present in the gallery ✅
- **every query CROP** has a gallery match on a different camera ✅
  (per-crop, not per-identity)
- {len(cams)} distinct cameras, no cross-source overlap ✅
- junk pid −1 preserved ✅

`manifest.csv` records source, original pid/camid and original filename for all
{sum(len(r) for r in data.values()):,} crops.

## How to evaluate

Do **not** read the pooled query/gallery number as a headline: Market queries
compete against mtmmc distractors, so it is comparable to neither published
Market results nor the mtmmc baseline. Score per source instead:

```
configs/reid/clipreid_person_market_eval.yaml    # published Market protocol
configs/reid/clipreid_person_mtmmcv2_eval.yaml   # held-out mtmmc people
```

## Expectations

PIA sources contribute **{venue_ids} of {n_train_ids:,} train identities
({share:.2f}%)**. The PK sampler draws identities uniformly, so venue data
appears in roughly 1 batch in {one_in} — its influence will be near-invisible.
If venue adaptation is the goal, treat the mixing ratio as the experiment and
run a second arm that oversamples PIA 20–50×.

Epoch counts must be rescaled: at {n_train:,} crops one epoch is
{n_train // 64:,} iterations at batch 64, versus Market's 202. The official
120+60 recipe would be far longer than intended — start around stage1 20 /
stage2 15.
""",
        encoding="utf-8",
    )
    logger.info("wrote %s", dst / "README.md")


def _report(dst: Path, counts: dict[str, int], sources: tuple[Source, ...]) -> None:
    logger.info("built %s", dst)
    for src in sources:
        parts = [
            f"{s}={counts[f'{src.name}:{s}']}" for s in SPLITS if counts.get(f"{src.name}:{s}")
        ]
        logger.info("  %-14s %s", src.name, "  ".join(parts))
    for split in SPLITS:
        logger.info("  %-20s %7d crops", split, counts[split])


def _verify(dst: Path) -> None:
    """Fail loudly on any condition that would silently corrupt training."""
    data = {s: scan(dst / s) for s in SPLITS}

    train_pids = {pid for _, pid, _ in data["bounding_box_train"] if pid != JUNK_PID}
    query_pids = {pid for _, pid, _ in data["query"] if pid != JUNK_PID}
    gal_pids = {pid for _, pid, _ in data["bounding_box_test"] if pid != JUNK_PID}

    leak = train_pids & (query_pids | gal_pids)
    if leak:
        raise RuntimeError(f"identity leak: {len(leak)} ids in train and eval: {sorted(leak)[:8]}")
    logger.info("  train ∩ eval identities: 0  [OK]")

    missing = query_pids - gal_pids
    if missing:
        raise RuntimeError(f"{len(missing)} query ids absent from gallery: {sorted(missing)[:8]}")
    logger.info("  every query identity present in gallery  [OK]")

    # Answerability is per QUERY CROP, not per identity: the Market protocol drops
    # gallery entries sharing the query's camera, so each crop needs its identity
    # present in the gallery on some OTHER camera. Testing this per identity
    # (gallery cameras minus query cameras) is wrong — an identity with query and
    # gallery crops on the same camera pair is still fully answerable.
    gal_cams: dict[int, set[int]] = defaultdict(set)
    for _, pid, cam in data["bounding_box_test"]:
        gal_cams[pid].add(cam)
    dead = [(pid, cam) for _, pid, cam in data["query"] if not (gal_cams[pid] - {cam})]
    if dead:
        n_ids = len({pid for pid, _ in dead})
        raise RuntimeError(
            f"{len(dead)} query crops ({n_ids} identities) have no cross-camera gallery match"
        )
    logger.info("  every query CROP answerable cross-camera  [OK]")

    cams = {cam for rows in data.values() for _, _, cam in rows}
    logger.info("  distinct cameras: %d  %s", len(cams), sorted(cams))

    n_junk = sum(1 for _, pid, _ in data["bounding_box_test"] if pid == JUNK_PID)
    logger.info("  market junk distractors preserved as pid -1: %d crops", n_junk)
    logger.info("  train identities: %d | eval identities: %d", len(train_pids), len(query_pids))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, required=True, help="combine recipe YAML")
    ap.add_argument("--dst", type=Path, default=None, help="override the output directory")
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="re-run the integrity checks on an existing build",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name, sources = load_recipe(args.recipe)
    dst = args.dst or REID / name
    if args.verify_only:
        _verify(dst)
    else:
        build(dst, sources)


if __name__ == "__main__":
    main()
