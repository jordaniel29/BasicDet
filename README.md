# basicdet — Person Detection & ReID

Repo for fine-tuning and evaluating the models used for multi-camera tracking:

- **Person detection** — YOLO26 (Ultralytics) and RF-DETR
- **Person ReID embedders** — `reid_ftnet` (layumi ft_net), `reid_clipreid`
  (official CLIP-ReID two-stage recipe) and `reid_personvit` (PersonViT
  ViT-B/16, TransReID fine-tune)

One YAML fully describes a run; its `family` field selects the pipeline, so a
single entry point works for every model. No hyperparameters live in the code.

## Install

Managed with **conda** (env name `persondet`), targeting **Python 3.11+**:

```bash
conda activate persondet
pip install -e ".[dev]"
```

## Usage

All three entry points are config-driven and run as modules:

```bash
python -m basicdet.train    --config configs/yolo/yolo26l_person_v6.3.yaml
python -m basicdet.evaluate  --config <config.yaml> --weights <checkpoint>
python -m basicdet.predict   --config <config.yaml> --weights <checkpoint> --source <images>
```

> Run as `python -m basicdet.train` (module form) — invoking `basicdet/train.py`
> by path breaks the package import. Long runs go in `tmux`.

The config's `family` key dispatches to the right pipeline:

| `family` | Model | Config dir |
|---|---|---|
| `yolo` | YOLO26 detection | `configs/yolo/` |
| `rfdetr` | RF-DETR detection | `configs/rfdetr/` |
| `reid_ftnet` | ft_net ReID embedder | `configs/reid/` |
| `reid_clipreid` | CLIP-ReID embedder | `configs/reid/` |
| `reid_personvit` | PersonViT ReID embedder | `configs/reid/` |

## Layout

Flat layout (BasicSR-style): the importable package sits at the repo root.

```text
basicdet/                 # importable package (pip-installed editable)
├── train.py  evaluate.py  predict.py   # config-driven entry points
├── models/               # per-family pipelines (yolo, rfdetr, reid_*)
├── metrics/              # COCO mAP, ReID mAP/CMC
└── utils/                # config schemas, registry, seed, tracking
configs/{yolo,rfdetr,reid}/*.yaml       # one YAML per experiment
curation/                 # dataset build / review / packaging tools
assets/data/              # datasets (gitignored)
runs/                     # training / eval outputs (gitignored)
tests/                    # pytest
```

Datasets and checkpoints are **not** committed (`assets/data/` and `runs/` are
gitignored). Experiment tracking uses **Weights & Biases** (project `person-det`).

## Development

Tooling is configured in `pyproject.toml` and enforced via pre-commit:

```bash
ruff format .        # format (line length 100)
ruff check --fix .   # lint + import sort
mypy basicdet/       # type check
pytest               # tests
```

See [CLAUDE.md](CLAUDE.md) for the full coding conventions and project-specific
guidance (dataset versioning, ReID deployment contracts, etc.).

## Adding a model

Add a `basicdet/models/<x>.py` exposing `train` / `evaluate` / `predict`, a
config schema in `basicdet/utils/config.py`, and one branch in
`basicdet/utils/registry.py`.
