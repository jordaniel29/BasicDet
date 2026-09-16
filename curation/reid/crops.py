"""Cut identity-labelled person crops out of tracked footage.

The inputs every ReID source has in common: a video per (scenario, camera) and a
MOT ground-truth file beside it whose track ids are consistent across the cameras
of that scenario. What differs per dataset — the scenario names, the frame rate,
which cameras exist — is passed in, not hardcoded.

Three quality floors are applied before a box is ever cut, because each one has
produced a bad dataset here before:

* **size** — a box smaller than the floors is upscaling noise at 256x128 input.
* **visibility** — MOT's ``vis`` column marks occlusion by another person. The
  first AI-Hub export ignored it and shipped crops that were mostly someone else.
* **frame edge** — a box touching the border is a half body walking in or out.
  ``vis`` does NOT catch this: it stays 1.0 while the body is simply outside the
  frame, which is why the edge margin is a separate, explicit floor.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("curation.reid.crops")

# Context kept around each box. Enough to show the silhouette's edges without
# pulling in a neighbouring person.
DEFAULT_PAD_FRAC = 0.08
JPEG_QUALITY = 95


@dataclass(frozen=True)
class MotBox:
    """One row of MOT ground truth: ``frame,id,left,top,w,h,conf,class,vis``.

    Attributes:
        frame: 1-based index of the decoded source frame.
        tid: Track id. Assumed consistent across the cameras of one scenario.
        x: Left edge in source-frame pixels.
        y: Top edge in source-frame pixels.
        w: Box width in pixels.
        h: Box height in pixels.
        vis: Visibility in ``[0, 1]``; 1.0 means unoccluded by another target.
    """

    frame: int
    tid: int
    x: float
    y: float
    w: float
    h: float
    vis: float


@dataclass(frozen=True)
class CropFloors:
    """Quality floors a box must clear to be cut.

    Attributes:
        min_w: Minimum box width in source pixels.
        min_h: Minimum box height in source pixels.
        min_visibility: Minimum MOT ``vis``; 0.0 disables the check.
        edge_margin_px: A box within this many pixels of any frame border is
            treated as truncated. 0 disables the check.
        frame_w: Source frame width, needed for the right-edge test.
        frame_h: Source frame height, needed for the bottom-edge test.
    """

    min_w: int = 32
    min_h: int = 80
    min_visibility: float = 0.65
    edge_margin_px: int = 2
    frame_w: int = 1920
    frame_h: int = 1080

    def accepts(self, box: MotBox) -> bool:
        """Whether ``box`` clears every floor."""
        if box.w < self.min_w or box.h < self.min_h:
            return False
        if box.vis < self.min_visibility:
            return False
        m = self.edge_margin_px
        if m <= 0:
            return True
        return not (
            box.x <= m
            or box.y <= m
            or box.x + box.w >= self.frame_w - m
            or box.y + box.h >= self.frame_h - m
        )


@dataclass(frozen=True)
class CropSpec:
    """One crop to cut: which identity, which view, which frame, which box.

    Attributes:
        pid: Identity id, as it will appear in the crop filename.
        camera: Physical camera id.
        sequence: Scenario/sequence number, kept in the name so a crop is
            traceable to its source video.
        box: The MOT box to cut.
    """

    pid: int
    camera: int
    sequence: int
    box: MotBox

    @property
    def name(self) -> str:
        """Crop filename in this repo's shared convention."""
        return crop_name(self.pid, self.camera, self.sequence, self.box.frame)


def crop_name(pid: int, camera: int, sequence: int, frame: int) -> str:
    """``<pid>_c<cam>_s<seq>_f<frame>.jpg`` — the naming every loader here parses.

    The Market-1501 parsers used by CLIP-ReID, PersonViT and ft_net read pid and
    camera id straight out of the name with ``(\\d+)_c(\\d+)``, so the leading two
    fields are load-bearing; the rest is provenance.
    """
    return f"{pid}_c{camera}_s{sequence:02d}_f{frame:06d}.jpg"


def load_mot(path: Path) -> list[MotBox]:
    """Parse one MOT ground-truth file, sorted by frame then track id.

    Args:
        path: File of ``frame,id,left,top,w,h,conf,class,vis`` rows.

    Returns:
        Every row as a :class:`MotBox`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On a row with fewer than 9 fields. That means a truncated or
            foreign file, which must fail loudly rather than be half-read.
    """
    if not path.is_file():
        raise FileNotFoundError(f"MOT label file not found: {path}")
    boxes: list[MotBox] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        f = line.split(",")
        if len(f) < 9:
            raise ValueError(f"{path}:{lineno}: expected 9 MOT fields, got {len(f)}")
        boxes.append(
            MotBox(
                frame=int(float(f[0])),
                tid=int(float(f[1])),
                x=float(f[2]),
                y=float(f[3]),
                w=float(f[4]),
                h=float(f[5]),
                vis=float(f[8]),
            )
        )
    boxes.sort(key=lambda b: (b.frame, b.tid))
    return boxes


def plan_crops(
    boxes: list[MotBox],
    camera: int,
    sequence: int,
    stride_frames: int,
    floors: CropFloors,
) -> list[CropSpec]:
    """Pick one acceptable box per identity per ``stride_frames``.

    The stride is measured from the last *kept* box, not from the last seen one,
    so an occluded or truncated stretch does not consume the budget — the next
    usable box after it is taken instead of the window being skipped.

    Args:
        boxes: Rows for one video, any order.
        camera: Camera id to stamp on the resulting specs.
        sequence: Scenario/sequence number to stamp on the resulting specs.
        stride_frames: Minimum frame gap between kept crops of one identity.
            1 keeps every acceptable box.
        floors: Quality floors.

    Returns:
        Specs in frame order.
    """
    last_kept: dict[int, int] = {}
    specs: list[CropSpec] = []
    for b in sorted(boxes, key=lambda b: (b.frame, b.tid)):
        if not floors.accepts(b):
            continue
        if b.frame - last_kept.get(b.tid, -(10**9)) < stride_frames:
            continue
        specs.append(CropSpec(pid=b.tid, camera=camera, sequence=sequence, box=b))
        last_kept[b.tid] = b.frame
    return specs


def cut_box(image: np.ndarray, box: MotBox, pad_frac: float = DEFAULT_PAD_FRAC) -> np.ndarray:
    """Crop ``box`` out of ``image`` with proportional context, clipped to bounds."""
    h, w = image.shape[:2]
    pw, ph = box.w * pad_frac, box.h * pad_frac
    y1, y2 = max(0, int(box.y - ph)), min(h, int(box.y + box.h + ph))
    x1, x2 = max(0, int(box.x - pw)), min(w, int(box.x + box.w + pw))
    return image[y1:y2, x1:x2]


def cut_from_video(
    video: Path,
    specs: list[CropSpec],
    out_dir: Path,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> list[CropSpec]:
    """Decode ``video`` once and write every planned crop.

    Frames are grabbed sequentially and only decoded where a crop is wanted,
    which is what makes a full-corpus cut cheap: seeking per crop would decode
    the same GOP repeatedly.

    Args:
        video: Source video.
        specs: Crops to cut, from :func:`plan_crops`. MOT frames are 1-based.
        out_dir: Destination directory; created if absent.
        pad_frac: Context padding as a fraction of box size.

    Returns:
        The specs actually written, in the input order.

    Raises:
        FileNotFoundError: If ``video`` cannot be opened.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[CropSpec]] = defaultdict(list)
    for s in specs:
        by_frame[s.box.frame].append(s)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    written: list[CropSpec] = []
    index = 0
    try:
        while cap.grab():
            index += 1  # MOT frames are 1-based
            if index not in by_frame:
                continue
            ok, image = cap.retrieve()
            if not ok:
                continue
            for s in by_frame[index]:
                crop = cut_box(image, s.box, pad_frac)
                if crop.size == 0:
                    continue
                cv2.imwrite(str(out_dir / s.name), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                written.append(s)
    finally:
        cap.release()
    logger.info("%s: %d/%d crops", video.name, len(written), len(specs))
    return written
