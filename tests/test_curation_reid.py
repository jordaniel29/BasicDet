"""Tests for the reusable ReID curation library (``curation.reid``).

These cover the decisions that silently produce a bad dataset if they are wrong:
which boxes are rejected, how the temporal stride is measured, how crops are
routed to splits, and that a leaky or unanswerable split is refused rather than
written. Dataset-specific recipes live in ``curation/recipes/`` and are not
tested here — they hold constants, not logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from curation.reid import crops, package
from curation.reid import review as scoring


def box(
    frame: int,
    tid: int = 1,
    x: float = 500,
    y: float = 300,
    w: float = 60,
    h: float = 150,
    vis: float = 1.0,
) -> crops.MotBox:
    return crops.MotBox(frame=frame, tid=tid, x=x, y=y, w=w, h=h, vis=vis)


FLOORS = crops.CropFloors()


# --- MOT parsing --------------------------------------------------------------
def test_load_mot_parses_and_sorts_by_frame(tmp_path: Path) -> None:
    p = tmp_path / "cam1.txt"
    p.write_text(
        "55,6,838.35,161.62,50.42,120.37,1,1,1.00\n54,6,836.63,160.93,49.34,119.65,1,1,0.50\n\n"
    )
    boxes = crops.load_mot(p)
    assert [b.frame for b in boxes] == [54, 55]
    assert boxes[0].tid == 6 and boxes[0].vis == 0.5
    assert boxes[1].w == pytest.approx(50.42)


def test_load_mot_rejects_short_rows(tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("54,6,836.63,160.93,49.34\n")
    with pytest.raises(ValueError, match="expected 9 MOT fields"):
        crops.load_mot(p)


def test_load_mot_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        crops.load_mot(tmp_path / "nope.txt")


# --- quality floors -----------------------------------------------------------
def test_acceptable_box_passes() -> None:
    assert FLOORS.accepts(box(1))


@pytest.mark.parametrize(
    "kw",
    [
        {"h": 79.0},  # below min_h
        {"w": 31.0},  # below min_w
        {"vis": 0.64},  # occluded by another person
        {"x": 0.0},  # touching the left border
        {"y": 1.0},  # touching the top border
        {"x": 1860.0},  # right edge at the border
        {"y": 930.0},  # bottom edge at the border
    ],
)
def test_unacceptable_boxes_are_rejected(kw: dict[str, float]) -> None:
    assert not FLOORS.accepts(box(1, **kw))


def test_edge_check_can_be_disabled() -> None:
    # A source whose labels are already clipped to the frame should not lose every
    # box that merely touches the border.
    assert crops.CropFloors(edge_margin_px=0).accepts(box(1, x=0.0))


# --- temporal subsampling -----------------------------------------------------
def test_stride_keeps_one_box_per_window_per_identity() -> None:
    boxes = [box(f, tid=1) for f in range(1, 100)] + [box(f, tid=2) for f in range(1, 100)]
    specs = crops.plan_crops(boxes, camera=1, sequence=1, stride_frames=30, floors=FLOORS)
    for tid in (1, 2):
        assert [s.box.frame for s in specs if s.pid == tid] == [1, 31, 61, 91]


def test_stride_is_measured_from_the_last_kept_box() -> None:
    # Frames 1-40 are occluded, so the first kept box is 41 and the next is 71 —
    # the unusable stretch must not consume the stride budget.
    boxes = [box(f, vis=0.1) for f in range(1, 41)] + [box(f) for f in range(41, 100)]
    specs = crops.plan_crops(boxes, camera=1, sequence=1, stride_frames=30, floors=FLOORS)
    assert [s.box.frame for s in specs] == [41, 71]


def test_stride_of_one_keeps_every_acceptable_box() -> None:
    boxes = [box(f) for f in range(1, 6)]
    specs = crops.plan_crops(boxes, camera=1, sequence=1, stride_frames=1, floors=FLOORS)
    assert len(specs) == 5


# --- naming -------------------------------------------------------------------
def test_crop_name_is_parseable_by_the_market_loaders() -> None:
    assert crops.crop_name(10, 4, 8, 1234) == "10_c4_s08_f001234.jpg"
    assert crops.CropSpec(pid=10, camera=4, sequence=8, box=box(1234, tid=10)).name == (
        "10_c4_s08_f001234.jpg"
    )


# --- review decisions ---------------------------------------------------------
def test_review_decisions_resolve() -> None:
    actions = {2: "drop", 3: "merge:17"}
    assert package.resolve_pid(2, actions) is None
    assert package.resolve_pid(3, actions) == 17
    assert package.resolve_pid(5, actions) == 5  # absent means keep


def test_unknown_review_action_fails_loudly() -> None:
    # A typo in a hand-edited decisions file must not silently become "keep".
    with pytest.raises(ValueError, match="unknown review action"):
        package.resolve_pid(2, {2: "kepe"})


def test_missing_decisions_file_keeps_everything(tmp_path: Path) -> None:
    assert package.load_decisions(tmp_path / "absent.csv") == {}


# --- split assignment ---------------------------------------------------------
def test_heldout_ids_go_to_query_or_gallery_by_camera() -> None:
    assert package.assign_split(8, 1, (8,), query_camera=1) == "query"
    assert package.assign_split(8, 3, (8,), query_camera=1) == "bounding_box_test"


def test_train_ids_never_reach_eval() -> None:
    for cam in (1, 3):
        assert package.assign_split(1, cam, (8,), query_camera=1) == "bounding_box_train"


# --- the two checks that make an eval number meaningful ------------------------
def _crop(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8\xff")  # never decoded by the packaging code
    return p


def test_leaky_split_is_refused(tmp_path: Path) -> None:
    src = _crop(tmp_path, "a.jpg")
    # A recipe rolling its own split rule (by scenario, by session) can put one
    # identity on both sides. That is the defect that made two earlier datasets
    # report memorisation as skill.
    rows = [
        (src, 1, 1, "1_c1_s01_f000001.jpg", "bounding_box_train"),
        (src, 1, 2, "1_c2_s01_f000001.jpg", "query"),
        (src, 1, 3, "1_c3_s01_f000001.jpg", "bounding_box_test"),
    ]
    with pytest.raises(RuntimeError, match="share 1 identities"):
        package.write_market_layout(tmp_path / "ds", rows)


def test_query_with_no_cross_camera_gallery_match_is_refused(tmp_path: Path) -> None:
    src = _crop(tmp_path, "b.jpg")
    # Identity 8's gallery crop is on the query camera, so the Market protocol's
    # same-camera filter leaves the query unanswerable.
    rows = [
        (src, 8, 1, "8_c1_s01_f000001.jpg", "query"),
        (src, 8, 1, "8_c1_s01_f000002.jpg", "bounding_box_test"),
    ]
    with pytest.raises(RuntimeError, match="no cross-camera gallery match"):
        package.write_market_layout(tmp_path / "ds", rows)


def test_unknown_split_name_fails_loudly(tmp_path: Path) -> None:
    src = _crop(tmp_path, "c.jpg")
    with pytest.raises(ValueError, match="unknown split"):
        package.write_market_layout(tmp_path / "ds", [(src, 1, 1, "1_c1.jpg", "val")])


def test_valid_split_is_written_and_counted(tmp_path: Path) -> None:
    src = _crop(tmp_path, "d.jpg")
    rows = [
        (src, 1, 1, "1_c1_s01_f000001.jpg", "bounding_box_train"),
        (src, 8, 1, "8_c1_s01_f000001.jpg", "query"),
        (src, 8, 2, "8_c2_s01_f000001.jpg", "bounding_box_test"),
    ]
    counts = package.write_market_layout(tmp_path / "ds", rows)
    assert counts == {"bounding_box_train": 1, "query": 1, "bounding_box_test": 1}
    assert (tmp_path / "ds/query/8_c1_s01_f000001.jpg").exists()


def test_packaging_is_idempotent(tmp_path: Path) -> None:
    src = _crop(tmp_path, "e.jpg")
    rows = [(src, 1, 1, "1_c1_s01_f000001.jpg", "bounding_box_train")]
    for _ in range(2):
        package.write_market_layout(tmp_path / "ds", rows)
    assert len(list((tmp_path / "ds/bounding_box_train").iterdir())) == 1


def test_assign_split_feeds_write_market_layout(tmp_path: Path) -> None:
    # The common identity-holdout rule must produce a split that passes the checks.
    src = _crop(tmp_path, "f.jpg")
    rows = [
        (src, pid, cam, f"{pid}_c{cam}.jpg", package.assign_split(pid, cam, (8,), 1))
        for pid, cam in [(1, 1), (1, 2), (8, 1), (8, 2)]
    ]
    counts = package.write_market_layout(tmp_path / "ds", rows)
    assert counts == {"bounding_box_train": 2, "query": 1, "bounding_box_test": 1}


# --- pruning ------------------------------------------------------------------
def test_prune_only_drops_scored_crops_below_a_positive_threshold() -> None:
    assert scoring.is_pruned(0.2, 0.45)
    assert not scoring.is_pruned(0.45, 0.45)
    assert not scoring.is_pruned(None, 0.45)  # unscored crops are never dropped silently
    assert not scoring.is_pruned(0.2, 0.0)  # pruning disabled


def test_crop_scores_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "review/crop_scores.csv"
    scoring.write_crop_scores(p, ["b.jpg", "a.jpg"], [0.5, 0.9])
    assert scoring.load_crop_scores(p) == {"a.jpg": 0.9, "b.jpg": 0.5}


def test_missing_crop_scores_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run the review stage"):
        scoring.load_crop_scores(tmp_path / "absent.csv")
