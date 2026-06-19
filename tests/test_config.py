"""Tests for experiment-config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from basicdet.utils.config import (
    RFDETRExperimentConfig,
    YOLOExperimentConfig,
    load_config,
    load_experiment,
    load_rfdetr_config,
)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_minimal_config_applies_defaults(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path / "exp.yaml", {"data": {"yaml_path": "data.yaml"}})

    config = load_config(cfg_path)

    assert config.data.yaml_path == Path("data.yaml")
    # Defaults fill in the rest of the schema.
    assert config.model.weights == "yolo26s.pt"
    assert config.train.epochs == 100
    assert config.train.seed == 42
    assert config.wandb.project == "person-det"


def test_load_overrides_nested_values(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "exp.yaml",
        {
            "data": {"yaml_path": "/abs/data.yaml"},
            "model": {"weights": "yolo26x.pt", "imgsz": 1280},
            "train": {"epochs": 5, "device": "cpu"},
            "wandb": {"enabled": False},
        },
    )

    config = load_config(cfg_path)

    assert config.model.imgsz == 1280
    assert config.train.epochs == 5
    assert config.train.device == "cpu"
    assert config.wandb.enabled is False


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_missing_required_data_section_raises(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path / "exp.yaml", {"model": {"weights": "yolo26n.pt"}})

    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_load_rfdetr_minimal_config_applies_defaults(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path / "rf.yaml", {"data": {"dataset_dir": "data/rfdetr"}})

    config = load_rfdetr_config(cfg_path)

    assert config.data.dataset_dir == Path("data/rfdetr")
    assert config.model.variant == "base"
    assert config.model.num_classes is None
    assert config.train.epochs == 50
    assert config.train.batch_size == 4
    assert config.train.extra == {}


def test_load_rfdetr_overrides_and_extra(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "rf.yaml",
        {
            "data": {"dataset_dir": "/abs/rfdetr"},
            "model": {"variant": "large", "resolution": 560},
            "train": {"epochs": 10, "extra": {"checkpoint_interval": 5}},
        },
    )

    config = load_rfdetr_config(cfg_path)

    assert config.model.variant == "large"
    assert config.model.resolution == 560
    assert config.train.epochs == 10
    assert config.train.extra == {"checkpoint_interval": 5}


def test_rfdetr_rejects_unknown_variant(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "rf.yaml",
        {"data": {"dataset_dir": "d"}, "model": {"variant": "huge"}},
    )

    with pytest.raises(ValidationError):
        load_rfdetr_config(cfg_path)


def test_load_experiment_dispatches_to_yolo(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "exp.yaml",
        {"family": "yolo", "data": {"yaml_path": "data.yaml"}},
    )

    config = load_experiment(cfg_path)

    assert isinstance(config, YOLOExperimentConfig)
    assert config.family == "yolo"


def test_load_experiment_dispatches_to_rfdetr(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "exp.yaml",
        {"family": "rfdetr", "data": {"dataset_dir": "rfdetr"}},
    )

    config = load_experiment(cfg_path)

    assert isinstance(config, RFDETRExperimentConfig)
    assert config.family == "rfdetr"


def test_load_experiment_rejects_unknown_family(tmp_path: Path) -> None:
    cfg_path = _write_yaml(tmp_path / "exp.yaml", {"family": "detr", "data": {}})

    with pytest.raises(ValidationError):
        load_experiment(cfg_path)
