"""Convert PersonViT's LUPerson MIM checkpoint into a TransReID-layout backbone.

``lakeAGI/PersonViT`` publishes the **pre-ReID** self-supervised checkpoints:
iBOT-style masked-image-modelling on LUPerson, no identity labels. They are the
weights the released ``lakeAGI/PersonViTReID`` checkpoints were then TransReID
fine-tuned from -- ``vitb.lup...n8/checkpoint0260.pth`` is the ``e0260`` in
``msmt.vitb.lup...e0260.transformer_120.pth``.

That makes the MIM checkpoint the right starting point for a "no ReID
pretraining" arm: it is to PersonViT what OpenAI CLIP is to CLIP-ReID -- general
representation learning, never trained on an identity objective.

The raw file is a 1.5 GB iBOT *training* checkpoint (student + teacher + iBOT
loss centers + optimizer state). ``basicdet.models.reid_personvit`` wants a
TransReID-layout state-dict, so this takes the **teacher** backbone -- the EMA
weights, which is what DINO/iBOT downstream transfer uses -- strips the
``backbone.`` prefix, re-prefixes it ``base.``, and drops everything else.

The result is backbone-only: no ``bottleneck.*``, so ``build_model`` initialises
the BNNeck fresh, and no ``classifier.*``, which it always reinitialises anyway.

Usage (from the repo root, conda env ``persondet``):
    python -m curation.convert_personvit_mim
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger("curation.personvit_mim")

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "lakeAGI/PersonViT"
HF_FILE = "vitb.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0260.pth"
DEFAULT_OUT = REPO_ROOT / "assets/model/reid/personvit_vitb_lup_mim_e0260_backbone.pth"

# Which sub-tree of the iBOT checkpoint to take. The teacher is the EMA of the
# student and is the standard choice for downstream transfer; the student also
# carries a DDP ``module.`` prefix and one extra tensor.
SOURCE_TREE = "teacher"
BACKBONE_PREFIX = "backbone."


def convert(src: Path, dst: Path) -> None:
    """Write the MIM teacher backbone as a TransReID ``base.*`` state-dict.

    Args:
        src: The downloaded iBOT training checkpoint.
        dst: Destination ``.pth``.

    Raises:
        FileNotFoundError: If ``src`` is missing.
        KeyError: If the checkpoint has no ``teacher`` tree.
        ValueError: If the teacher carries no ``backbone.`` tensors.
    """
    if not src.is_file():
        raise FileNotFoundError(f"MIM checkpoint not found: {src}")
    # weights_only=False: the file stores an argparse.Namespace under "args".
    raw = torch.load(src, map_location="cpu", weights_only=False)
    if SOURCE_TREE not in raw:
        raise KeyError(f"{src.name} has no {SOURCE_TREE!r} tree — keys: {list(raw)}")

    backbone = {
        f"base.{k.removeprefix(BACKBONE_PREFIX)}": v
        for k, v in raw[SOURCE_TREE].items()
        if k.startswith(BACKBONE_PREFIX)
    }
    if not backbone:
        raise ValueError(f"no {BACKBONE_PREFIX!r} tensors under {SOURCE_TREE!r} in {src.name}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone, dst)
    pos = backbone.get("base.pos_embed")
    logger.info(
        "wrote %s — %d tensors, pos_embed=%s (MIM epoch %s)",
        dst,
        len(backbone),
        tuple(pos.shape) if pos is not None else "?",
        raw.get("epoch", "?"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help=f"local iBOT checkpoint; default downloads {HF_REPO}/{HF_FILE}",
    )
    ap.add_argument("--dst", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src = args.src
    if src is None:
        from huggingface_hub import hf_hub_download

        logger.info("downloading %s/%s (~1.5 GB, cached by huggingface_hub)", HF_REPO, HF_FILE)
        src = Path(hf_hub_download(HF_REPO, HF_FILE))
    convert(src, args.dst)


if __name__ == "__main__":
    main()
