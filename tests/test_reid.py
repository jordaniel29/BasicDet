"""Tests for the ReID pipelines: metrics, data layout, checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from basicdet.metrics.reid import cosine_distance, evaluate_retrieval
from basicdet.models.reid_data import (
    RandomIdentitySampler,
    ReIDSplit,
    load_market_dataset,
    parse_market_name,
)
from basicdet.models.reid_ftnet import FtNet, _strip_head_for_inference
from basicdet.utils.config import load_experiment

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_retrieval_perfect_features_scores_one() -> None:
    # 3 identities, orthogonal features: retrieval must be perfect.
    eye = np.eye(3, dtype=np.float32)
    q_pids = np.array([0, 1, 2])
    q_cams = np.array([0, 0, 0])
    g_feats = np.concatenate([eye, eye])  # each identity twice, other cameras
    g_pids = np.array([0, 1, 2, 0, 1, 2])
    g_cams = np.array([1, 1, 1, 2, 2, 2])
    res = evaluate_retrieval(eye, q_pids, q_cams, g_feats, g_pids, g_cams)
    assert res["mAP"] == pytest.approx(1.0)
    assert res["rank1"] == pytest.approx(1.0)


def test_retrieval_same_camera_matches_are_excluded() -> None:
    # The only same-identity gallery entry shares the query camera -> query is
    # dropped per protocol, and with no valid queries left the eval must raise.
    q = np.eye(2, dtype=np.float32)[:1]
    g = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="cross-camera"):
        evaluate_retrieval(
            q,
            np.array([5]),
            np.array([3]),
            g,
            np.array([5, 9]),
            np.array([3, 1]),
        )


def test_retrieval_known_answer_map() -> None:
    # One query; ranked gallery = [wrong, right, right] -> AP = (1/2 + 2/3) / 2.
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[0.99, 0.141], [0.98, 0.2], [0.97, 0.24]], dtype=np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    res = evaluate_retrieval(
        q,
        np.array([1]),
        np.array([0]),
        g,
        np.array([2, 1, 1]),
        np.array([1, 1, 1]),
        ranks=(1, 5),
    )
    assert res["mAP"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert res["rank1"] == pytest.approx(0.0)
    assert res["rank5"] == pytest.approx(1.0)


def test_retrieval_miss_outside_max_rank_scores_zero() -> None:
    # First correct match at rank 3 with max_rank=2: rank1 and rank2 must be 0
    # (regression: a leftover line used to credit the last rank unconditionally).
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[0.99, 0.141], [0.98, 0.2], [0.5, 0.866]], dtype=np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    res = evaluate_retrieval(
        q,
        np.array([1]),
        np.array([0]),
        g,
        np.array([2, 2, 1]),
        np.array([1, 1, 1]),
        ranks=(1, 2),
    )
    assert res["rank1"] == pytest.approx(0.0)
    assert res["rank2"] == pytest.approx(0.0)


def test_cosine_distance_range() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = cosine_distance(a, a)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Market-1501 layout
# --------------------------------------------------------------------------- #
def _make_market_dir(root: Path, n_ids: int = 3, cams: tuple[int, ...] = (1, 2)) -> Path:
    for sub in ("bounding_box_train", "query", "bounding_box_test"):
        (root / sub).mkdir(parents=True)
    img = Image.new("RGB", (64, 128), color=(120, 30, 200))
    for pid in range(n_ids):
        for cam in cams:
            img.save(root / "bounding_box_train" / f"{pid + 100:04d}_c{cam}_f000001.jpg")
            img.save(root / "query" / f"{pid + 500:04d}_c{cam}_f000002.jpg")
            img.save(root / "bounding_box_test" / f"{pid + 500:04d}_c{3 - cam}_f000003.jpg")
    img.save(root / "bounding_box_train" / "-1_c1_f000009.jpg")  # junk: must be dropped
    return root


def test_parse_market_name_variants() -> None:
    assert parse_market_name(Path("0042_c15_f0001250.jpg")) == (42, 15)
    assert parse_market_name(Path("-1_c3s2_012345_00.jpg")) == (-1, 3)
    with pytest.raises(ValueError):
        parse_market_name(Path("not_a_market_name.jpg"))


def test_load_market_dataset(tmp_path: Path) -> None:
    data = load_market_dataset(_make_market_dir(tmp_path))
    assert data.num_train_pids == 3
    assert len(data.train.paths) == 6  # junk -1 image dropped
    assert set(data.train.pids.tolist()) == {0, 1, 2}  # relabelled contiguous
    assert len(data.query.paths) == 6
    assert len(data.gallery.paths) == 6
    assert data.query.pids.min() == 500  # query/gallery keep original pids


def test_load_market_dataset_missing_split(tmp_path: Path) -> None:
    (tmp_path / "bounding_box_train").mkdir()
    with pytest.raises(FileNotFoundError, match="query"):
        load_market_dataset(tmp_path)


# --------------------------------------------------------------------------- #
# ft_net checkpoint compatibility with the TRACE loader
# --------------------------------------------------------------------------- #
class _TraceClassBlock(nn.Module):
    """Vendored replica of TRACE piapf/reid/ftnet_reid.py::_ClassBlock."""

    def __init__(self, input_dim: int, class_num: int, linear: int = 512) -> None:
        super().__init__()
        block: list[nn.Module] = []
        if linear > 0:
            block.append(nn.Linear(input_dim, linear))
        else:
            linear = input_dim
        block.append(nn.BatchNorm1d(linear))
        self.add_block = nn.Sequential(*block)
        self.classifier = nn.Sequential(nn.Linear(linear, class_num))


class _TraceFtNet(nn.Module):
    """Vendored replica of TRACE piapf/reid/ftnet_reid.py::_FtNet."""

    def __init__(self, class_num: int = 751, stride: int = 2, linear_num: int = 512) -> None:
        super().__init__()
        from torchvision import models

        backbone = models.resnet50(weights=None)
        if stride == 1:
            backbone.layer4[0].downsample[0].stride = (1, 1)
            backbone.layer4[0].conv2.stride = (1, 1)
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = backbone
        self.classifier = _TraceClassBlock(2048, class_num, linear=linear_num)


def test_ftnet_checkpoint_loads_into_trace_loader(tmp_path: Path) -> None:
    """A trainer checkpoint must load into TRACE's _FtNet with no real misses.

    Replicates the TRACE load procedure exactly: strip ``classifier.classifier.*``
    (the head is dropped at inference), then ``strict=False`` — anything left
    missing/unexpected would mean a bottleneck/backbone schema drift.
    """
    ours = FtNet(class_num=17, stride=2, linear_num=512, droprate=0.5, imagenet_init=False)
    ckpt = tmp_path / "net_last.pth"
    torch.save(ours.state_dict(), ckpt)

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    head_keys = [k for k in state if k.startswith("classifier.classifier.")]
    for k in head_keys:
        del state[k]
    theirs = _TraceFtNet(class_num=999, stride=2, linear_num=512)
    missing, unexpected = theirs.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if not m.startswith("classifier.classifier.")]
    assert head_keys, "trainer checkpoint lost its class head"
    assert real_missing == [], f"schema drift vs TRACE loader: missing {real_missing[:5]}"
    assert list(unexpected) == [], (
        f"schema drift vs TRACE loader: unexpected {list(unexpected)[:5]}"
    )


def test_ftnet_inference_feature_shape() -> None:
    model = FtNet(class_num=5, stride=2, linear_num=512, droprate=0.0, imagenet_init=False)
    model = _strip_head_for_inference(model).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 256, 128))
    assert out.shape == (2, 512)


# --------------------------------------------------------------------------- #
# P x K identity sampler
# --------------------------------------------------------------------------- #
def _split(pids: list[int]) -> ReIDSplit:
    return ReIDSplit(
        [Path(f"{pid}_c0_{i}.jpg") for i, pid in enumerate(pids)],
        np.asarray(pids, dtype=np.int64),
        np.zeros(len(pids), dtype=np.int64),
    )


def test_pk_sampler_batches_hold_k_crops_per_identity() -> None:
    # 6 identities x 8 crops; batch 8 = 2 identities x 4 crops.
    split = _split([pid for pid in range(6) for _ in range(8)])
    order = list(RandomIdentitySampler(split, batch_size=8, num_instances=4))
    assert len(order) % 8 == 0
    for start in range(0, len(order), 8):
        counts = np.unique(split.pids[order[start : start + 8]], return_counts=True)[1]
        # Exactly 2 identities, 4 crops each — what batch-hard mining needs.
        assert sorted(counts.tolist()) == [4, 4]


def test_pk_sampler_oversamples_short_identities() -> None:
    # Identity 1 has a single crop; it must still appear K times when drawn.
    split = _split([0, 0, 0, 0, 0, 0, 0, 0, 1])
    order = list(RandomIdentitySampler(split, batch_size=8, num_instances=4))
    assert len(order) >= 8
    assert len(order) % 4 == 0


def test_pk_sampler_rejects_impossible_batch_shape() -> None:
    split = _split([0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="multiple of num_instances"):
        RandomIdentitySampler(split, batch_size=10, num_instances=4)
    with pytest.raises(ValueError, match="identities per batch"):
        RandomIdentitySampler(split, batch_size=64, num_instances=4)


# --------------------------------------------------------------------------- #
# PersonViT — losses and LR schedule (deterministic logic)
# --------------------------------------------------------------------------- #
def test_triplet_loss_zero_when_positives_coincide() -> None:
    from basicdet.models.reid_personvit import batch_hard_triplet_loss

    # Identical positives, far negatives: d_ap = 0, d_an large -> soft margin ~ 0.
    feats = torch.tensor([[0.0, 0.0], [0.0, 0.0], [50.0, 0.0], [50.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    assert float(batch_hard_triplet_loss(feats, labels)) == pytest.approx(0.0, abs=1e-6)


def test_triplet_loss_hinge_margin_matches_hand_computation() -> None:
    from basicdet.models.reid_personvit import batch_hard_triplet_loss

    # 1-d features, two identities: {0, 1} and {3, 4}. Per anchor, mining gives
    # (d_ap, d_an) = (1, 3), (1, 2), (1, 2), (1, 3) and the loss is
    # mean(max(0, d_ap - d_an + margin)).
    feats = torch.tensor([[0.0], [1.0], [3.0], [4.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    # margin 0.3: every anchor already clears it -> 0.
    assert float(batch_hard_triplet_loss(feats, labels, margin=0.3)) == pytest.approx(0.0)
    # margin 2.0: the two inner anchors violate it by 1 each -> mean 0.5.
    assert float(batch_hard_triplet_loss(feats, labels, margin=2.0)) == pytest.approx(0.5)


def test_triplet_loss_requires_pk_batch() -> None:
    from basicdet.models.reid_personvit import batch_hard_triplet_loss

    feats = torch.zeros(4, 8)
    with pytest.raises(ValueError, match="num_instances"):
        batch_hard_triplet_loss(feats, torch.tensor([0, 1, 2, 3]))


def test_lr_schedule_warms_up_then_cosine_decays() -> None:
    from basicdet.models.reid_personvit import lr_at_epoch

    base, epochs, warmup = 4e-4, 120, 20
    at = [lr_at_epoch(e, base, epochs, warmup) for e in range(epochs + 1)]
    assert at[0] == pytest.approx(0.01 * base)  # timm warmup_lr_init
    assert at[warmup - 1] < base  # warm-up is still climbing at its last epoch
    assert all(b > a for a, b in zip(at[: warmup - 1], at[1:warmup], strict=True))
    # After warm-up the cosine takes over and decays monotonically to lr_min.
    assert all(b < a for a, b in zip(at[warmup:-2], at[warmup + 1 : -1], strict=True))
    assert at[epochs] == pytest.approx(0.002 * base)


# --------------------------------------------------------------------------- #
# PersonViT checkpoint compatibility with the TRACE loader
# --------------------------------------------------------------------------- #
# Tiny synthetic ViT: 32x16 crops, 8-px patches -> 4x2 = 8 patches + class token.
_PV_HW, _PV_PATCH, _PV_DIM, _PV_DEPTH = (32, 16), 8, 64, 2


def _tiny_personvit_checkpoint(path: Path, num_ids: int = 7) -> None:
    """Write a released-checkpoint-shaped .pth for a 2-block, 64-d ViT."""
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(0)
    vit = VisionTransformer(
        img_size=_PV_HW,
        patch_size=_PV_PATCH,
        embed_dim=_PV_DIM,
        depth=_PV_DEPTH,
        num_heads=_PV_DIM // 64,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=0,
        class_token=True,
        global_pool="token",
    )
    state = {f"base.{k}": v for k, v in vit.state_dict().items()}
    state.update({f"bottleneck.{k}": v for k, v in nn.BatchNorm1d(_PV_DIM).state_dict().items()})
    state["classifier.weight"] = torch.zeros(num_ids, _PV_DIM)
    # TransReID's unused ImageNet head — present in the real released weights.
    state["base.fc.weight"] = torch.zeros(1000, _PV_DIM)
    state["base.fc.bias"] = torch.zeros(1000)
    torch.save(state, path)


def _trace_personvit_load(weights: Path, input_hw: tuple[int, int]) -> tuple[list[str], list[str]]:
    """Replica of TRACE apps/trace/worker/reid/personvit.py::PersonViTReIDEmbedder._build.

    Returns the (missing, unexpected) keys its ``load_state_dict`` reports — the
    real adapter raises unless both are empty.
    """
    from timm.models.vision_transformer import VisionTransformer

    state = torch.load(weights, map_location="cpu", weights_only=True)
    backbone, bn = {}, {}
    for k, v in state.items():
        if k.startswith(("classifier.", "base.fc.", "base.head.")):
            continue
        if k.startswith("bottleneck."):
            bn[k[len("bottleneck.") :]] = v
        elif k.startswith("base."):
            backbone[k[len("base.") :]] = v
        else:
            raise KeyError(f"PersonViT: unexpected checkpoint key {k!r}")
    embed_dim = int(backbone["cls_token"].shape[-1])
    depth = 1 + max(int(k.split(".")[1]) for k in backbone if k.startswith("blocks."))
    patch = int(backbone["patch_embed.proj.weight"].shape[-1])
    n_tokens = int(backbone["pos_embed"].shape[1])
    assert n_tokens == 1 + (input_hw[0] // patch) * (input_hw[1] // patch)
    model = VisionTransformer(
        img_size=input_hw,
        patch_size=patch,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=embed_dim // 64,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=0,
        class_token=True,
        global_pool="token",
    )
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    nn.BatchNorm1d(embed_dim).load_state_dict(bn)  # raises on a BNNeck shape drift
    return list(missing), list(unexpected)


def test_personvit_checkpoint_loads_into_trace_loader(tmp_path: Path) -> None:
    """A trainer checkpoint must load into TRACE's personvit adapter unchanged.

    The adapter raises on ANY missing/unexpected backbone key and on any key
    outside base./bottleneck./classifier., so this is the deployment contract:
    if it drifts, `backend: personvit` cannot load what we train.
    """
    from basicdet.models.reid_personvit import build_model

    source = tmp_path / "released.pth"
    _tiny_personvit_checkpoint(source, num_ids=7)
    model = build_model(source, num_classes=3, input_size=_PV_HW, drop_path_rate=0.1)
    ours = tmp_path / "transformer_last.pth"
    torch.save(model.state_dict(), ours)

    state = torch.load(ours, map_location="cpu", weights_only=True)
    assert "classifier.weight" in state, "trainer checkpoint lost its identity head"
    assert not any(k.startswith("base.fc.") for k in state), "ImageNet head leaked into the ckpt"
    missing, unexpected = _trace_personvit_load(ours, _PV_HW)
    assert missing == [], f"schema drift vs TRACE loader: missing {missing[:5]}"
    assert unexpected == [], f"schema drift vs TRACE loader: unexpected {unexpected[:5]}"


def test_personvit_warm_start_preserves_backbone_and_bnneck(tmp_path: Path) -> None:
    from basicdet.models.reid_personvit import build_model

    source = tmp_path / "released.pth"
    _tiny_personvit_checkpoint(source, num_ids=7)
    state = torch.load(source, map_location="cpu", weights_only=True)
    model = build_model(source, num_classes=3, input_size=_PV_HW)

    assert torch.equal(model.base.cls_token, state["base.cls_token"])
    assert torch.equal(model.bottleneck.weight, state["bottleneck.weight"])
    # The head is re-sized to OUR identity count, not the source checkpoint's.
    assert model.classifier.weight.shape == (3, _PV_DIM)
    # BNNeck shift stays frozen at 0 (bag-of-tricks); everything else trains.
    assert not model.bottleneck.bias.requires_grad


def test_personvit_rejects_wrong_input_resolution(tmp_path: Path) -> None:
    from basicdet.models.reid_personvit import build_model

    source = tmp_path / "released.pth"
    _tiny_personvit_checkpoint(source)
    with pytest.raises(ValueError, match="position-embedding tokens"):
        build_model(source, num_classes=3, input_size=(64, 32))


def test_personvit_rejects_sie_checkpoint(tmp_path: Path) -> None:
    from basicdet.models.reid_personvit import build_model

    source = tmp_path / "sie.pth"
    _tiny_personvit_checkpoint(source)
    state = torch.load(source, map_location="cpu", weights_only=True)
    state["base.sie_embed"] = torch.zeros(3, 1, _PV_DIM)
    torch.save(state, source)
    with pytest.raises(ValueError, match="SIE"):
        build_model(source, num_classes=3, input_size=_PV_HW)


def test_personvit_forward_shapes_and_neck_feat(tmp_path: Path) -> None:
    from basicdet.models.reid_personvit import build_model

    source = tmp_path / "released.pth"
    _tiny_personvit_checkpoint(source)
    model = build_model(source, num_classes=5, input_size=_PV_HW, drop_path_rate=0.0)
    images = torch.randn(4, 3, *_PV_HW)

    model.train()
    logits, pre_bn = model(images)
    assert logits.shape == (4, 5)
    assert pre_bn.shape == (4, _PV_DIM)  # triplet loss consumes the pre-BN feature

    model.eval()
    with torch.no_grad():
        after = model(images)
        model.neck_feat = "before"
        before = model(images)
    assert after.shape == before.shape == (4, _PV_DIM)
    # The two read-out points must actually differ — the whole point of the
    # 2026-08-27 A/B (post-BN is what TRACE deploys).
    assert not torch.allclose(after, before)


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("path", "family"),
    [
        ("configs/reid/ftnet_person_v1.yaml", "reid_ftnet"),
        ("configs/reid/clipreid_person_v1.yaml", "reid_clipreid"),
        ("configs/reid/personvit_person_atustc9.yaml", "reid_personvit"),
    ],
)
def test_reid_configs_load(path: str, family: str) -> None:
    config = load_experiment(REPO / path)
    assert config.family == family
    assert config.model.input_size == (256, 128)  # must match TRACE inference
