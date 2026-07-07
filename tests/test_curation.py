"""Tests for the deterministic curation helpers (box math, parsing, splitting)."""

from __future__ import annotations

from pathlib import Path

import pytest

from curation import mtmdc
from curation.build_version import FrameRecord, assign_splits, is_test, select_stride


# --- naming round-trip --------------------------------------------------------
def test_frame_stem_parse_roundtrip() -> None:
    stem = mtmdc.frame_stem(1, 15, 150)
    assert stem == "mtmdc_s01_c15_000150"
    assert mtmdc.parse_stem(stem) == (1, 15, 150)


def test_parse_stem_rejects_bad_name() -> None:
    with pytest.raises(ValueError):
        mtmdc.parse_stem("wn_set1_video1_1_000000")


# --- box geometry -------------------------------------------------------------
def test_box_to_yolo_normalises_center_and_size() -> None:
    cx, cy, w, h = mtmdc.box_to_yolo((10, 20, 30, 60), width=100, height=200)
    assert (cx, cy, w, h) == (0.2, 0.2, 0.2, 0.2)


def test_box_to_coco_is_integer_xywh_and_area() -> None:
    bbox, area = mtmdc.box_to_coco((10, 20, 30, 60))
    assert bbox == [10, 20, 20, 40]
    assert area == 800


def test_clamp_box_clips_to_bounds() -> None:
    assert mtmdc.clamp_box((-5, -5, 50, 50), width=40, height=40) == (0.0, 0.0, 40.0, 40.0)


def test_clean_boxes_drops_degenerate_and_clamps() -> None:
    raw = [
        (10, 10, 30, 40),  # valid
        (0, 0, 0.5, 100),  # width <= 1px -> dropped
        (-5, -5, 1000, 1000),  # clamped to image bounds, still valid
    ]
    cleaned = mtmdc.clean_boxes(raw, width=200, height=200)
    assert (10, 10, 30, 40) in cleaned
    assert (0.0, 0.0, 200.0, 200.0) in cleaned
    assert len(cleaned) == 2


# --- VATIC parsing ------------------------------------------------------------
def test_parse_vatic_groups_by_frame(tmp_path: Path) -> None:
    txt = tmp_path / "camera01.txt"
    txt.write_text(
        '1 10 20 30 60 5 0 0 0 "PERSON"\n'
        '2 11 21 31 61 5 0 1 0 "PERSON"\n'
        '1 12 22 32 62 7 0 0 0 "PERSON"\n'
    )
    boxes = mtmdc.parse_vatic(txt)
    assert set(boxes) == {5, 7}
    assert boxes[5] == [(10, 20, 30, 60), (11, 21, 31, 61)]
    assert boxes[7] == [(12, 22, 32, 62)]


# --- split assignment ---------------------------------------------------------
def _records(scenarios: list[int], cameras: list[int], frames: int) -> list[FrameRecord]:
    out = []
    for s in scenarios:
        for c in cameras:
            for f in range(frames):
                out.append(
                    FrameRecord(s, c, f * mtmdc.SUBSAMPLE_EVERY, 1920, 1080, f"s{s}/c{c}/{f}.jpg")
                )
    return out


def test_is_test_rules() -> None:
    # v4.1: cameras 15-16 are test.
    assert is_test("v4.1", FrameRecord(1, 15, 0, 1920, 1080, "x"))
    assert not is_test("v4.1", FrameRecord(1, 14, 0, 1920, 1080, "x"))
    # v4.2: scenarios 18/19/42/43 are test.
    assert is_test("v4.2", FrameRecord(18, 1, 0, 1920, 1080, "x"))
    assert not is_test("v4.2", FrameRecord(17, 1, 0, 1920, 1080, "x"))


def test_v41_split_is_camera_disjoint_and_proportioned() -> None:
    recs = _records(scenarios=[1, 10], cameras=list(range(1, 17)), frames=10)
    splits = assign_splits("v4.1", recs)
    test_cams = {r.camera for r in splits["test"]}
    pool_cams = {r.camera for r in splits["train"]} | {r.camera for r in splits["val"]}
    assert test_cams == {15, 16}
    assert pool_cams == set(range(1, 15))
    pool_n = len(splits["train"]) + len(splits["val"])
    assert len(splits["val"]) == round(pool_n * mtmdc.VAL_FRACTION)


def test_v42_split_is_scenario_disjoint() -> None:
    recs = _records(scenarios=[17, 18, 19], cameras=[1, 2], frames=10)
    splits = assign_splits("v4.2", recs)
    assert {r.scenario for r in splits["test"]} == {18, 19}
    pool_scenarios = {r.scenario for r in splits["train"]} | {r.scenario for r in splits["val"]}
    assert pool_scenarios == {17}


def test_split_is_deterministic() -> None:
    recs = _records(scenarios=[1], cameras=list(range(1, 17)), frames=8)
    a = assign_splits("v4.1", recs)
    b = assign_splits("v4.1", recs)
    assert [r.stem for r in a["train"]] == [r.stem for r in b["train"]]
    assert [r.stem for r in a["val"]] == [r.stem for r in b["val"]]


def test_unknown_version_raises() -> None:
    with pytest.raises(ValueError):
        is_test("v9.9", FrameRecord(1, 1, 0, 1920, 1080, "x"))


# --- stride selection ---------------------------------------------------------
def test_select_stride_keeps_multiples() -> None:
    # Pool frames are multiples of SUBSAMPLE_EVERY (15): 0,15,30,45,60,...
    recs = _records(scenarios=[1], cameras=[1], frames=8)  # frames 0..105 step 15
    kept = select_stride(recs, stride=30)
    assert [r.frame for r in kept] == [0, 30, 60, 90]
    # stride == pool stride keeps everything.
    assert len(select_stride(recs, stride=mtmdc.SUBSAMPLE_EVERY)) == len(recs)


def test_select_stride_rejects_non_multiple() -> None:
    recs = _records(scenarios=[1], cameras=[1], frames=4)
    with pytest.raises(ValueError):
        select_stride(recs, stride=20)  # not a multiple of 15
    with pytest.raises(ValueError):
        select_stride(recs, stride=0)
