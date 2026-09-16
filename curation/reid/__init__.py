"""Reusable ReID dataset curation: cut crops, review them, package, combine, audit.

These modules are the *tools*. Everything specific to one dataset — which videos,
which scenarios, which identities are held out — belongs in a module under
``curation/recipes/`` (gitignored) or a combine recipe YAML, not here.

Building a new ReID source is four steps, each one call:

1. :mod:`curation.reid.crops`   — parse labels, apply quality floors, subsample in
   time, cut the crops out of the footage.
2. :mod:`curation.reid.review`  — embed with the deployed encoder, score every crop
   against its own identity centroid, render contact sheets for a human.
3. :mod:`curation.reid.package` — write the Market-1501 layout as hardlinks with an
   identity-disjoint holdout, refusing to write a leaky split.
4. :mod:`curation.reid.audit`   — structural audit of the result.

:mod:`curation.reid.combine` unions several packaged sources into one training set.
"""

from curation.reid.crops import (
    CropFloors,
    CropSpec,
    MotBox,
    crop_name,
    cut_from_video,
    load_mot,
    plan_crops,
)
from curation.reid.package import (
    SPLITS,
    assign_split,
    load_decisions,
    resolve_pid,
    write_market_layout,
)

__all__ = [
    "SPLITS",
    "CropFloors",
    "CropSpec",
    "MotBox",
    "assign_split",
    "crop_name",
    "cut_from_video",
    "load_decisions",
    "load_mot",
    "plan_crops",
    "resolve_pid",
    "write_market_layout",
]
