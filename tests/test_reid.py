"""Tests for the ReID pipelines: metrics, data layout, checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from basicdet.metrics.reid import cosine_distance, evaluate_retrieval
from basicdet.models.reid_data import load_market_dataset, parse_market_name
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
# configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("path", "family"),
    [
        ("configs/reid/ftnet_person_v1.yaml", "reid_ftnet"),
        ("configs/reid/clipreid_person_v1.yaml", "reid_clipreid"),
    ],
)
def test_reid_configs_load(path: str, family: str) -> None:
    config = load_experiment(REPO / path)
    assert config.family == family
    assert config.model.input_size == (256, 128)  # must match TRACE inference
