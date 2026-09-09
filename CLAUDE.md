# CLAUDE.md

Guidance for writing and reviewing Python in this repository. The goal is code
that is **simple, readable, modular, and reproducible**.

> **Reusing this file:** The coding & commenting conventions (§1–§2, §4–§5) are
> general-purpose. The repository structure (§3) and "Project-specific guidance"
> reflect *this* repo's flat layout. A portable, `src/`-layout master copy lives
> at `~/.claude/templates/python-CLAUDE.md` — start new projects from there.

## Python version & tooling

- Target **Python 3.11+**. Use modern syntax (`match`, `|` union types, etc.).
- Always work inside a **dedicated virtual environment**; pin dependencies. The
  env manager is a per-project choice (see project-specific guidance below).
- Formatting and linting are **non-negotiable and automated** — do not hand-format:
  - **`ruff format`** for formatting (line length 100).
  - **`ruff check --fix`** for linting and import sorting.
  - **`mypy`** (or `pyright`) for type checking.
- Configure all tooling in `pyproject.toml`.

## Core principles

1. **Readability over cleverness.** Code is read far more than written. Prefer
   the obvious solution.
2. **Functions do one thing.** Keep them short and named for what they do. If you
   need "and" to describe a function, split it.
3. **Separation of concerns.** Decouple distinct responsibilities into separate
   modules (Single Responsibility Principle).
4. **Fail loud, fail early.** Validate inputs at boundaries. Raise specific
   exceptions with actionable messages. Never silently swallow errors.
5. **No magic numbers / strings.** Constants, thresholds, and paths belong in
   config or named constants — never inlined in logic.
6. **Don't over-engineer.** Don't add a layer of indirection until there are at
   least two concrete callers. Write the simplest thing that works.

## 1. Coding conventions

### Type hinting

- **All functions have full type hints** (PEP 484) — parameters and return values.
- Prefer precise types: `list[int]`, `dict[str, float]`, `pathlib.Path` for
  filesystem paths. Avoid `Any`; if a type is genuinely dynamic, document why.
- Use `pathlib.Path` for all filesystem work; never string concatenation for paths.
- Use dataclasses or **Pydantic** models for structured data and config rather
  than passing around loose dicts.

### Structure & style

- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_SNAKE` for constants. Names should be descriptive — `batch_size`, not `bs`.
- **Imports:** stdlib, third-party, then local — grouped and sorted (ruff). No
  wildcard imports, no unused imports.
- **Line length:** 100 chars.
- **Avoid deep nesting.** Use early returns / guard clauses.
- **No dead code.** Delete commented-out blocks; git remembers them.

### Configuration as code

- **No magic numbers or hardcoded paths in scripts.** Parameters, thresholds, and
  paths live in **configuration files (YAML / TOML)**, not inlined in logic.
  (Twelve-Factor: store config separately.)
- Prefer typed config (Pydantic / dataclass) loaded from file, with sane defaults.
- Scripts accept config paths / overrides via CLI rather than editing source.
- Separate config from secrets; never commit credentials, API keys, or `.env`.

### Error handling & logging

- Use the **`logging`** module, not `print`, for anything beyond throwaway
  scripts. Configure once at the entrypoint.
- Catch specific exceptions, not bare `except:`. Add context before re-raising.
- Validate file existence, types, and shapes at boundaries with clear messages.

## 2. Commenting conventions

### Explain the "why", not the "what"

- Assume the reader understands Python. **Comments explain why an engineering
  choice was made**, not what the syntax does.
  - Bad: `# increment counter by 1`
  - Good: `# retry up to 3x — the upstream API drops ~1% of requests under load.`

### Google-style docstrings

- **Every class and public method has a docstring** detailing `Args`, `Returns`,
  and `Raises` (Google Python Style Guide — parses cleanly into Sphinx).

### Citations

- For non-obvious algorithms or custom implementations, **link the relevant
  paper or documentation** in the docstring.

### Standardized team tags

- Format actionable comments so they are easy to parse in PRs / peer reviews:
  - `TODO(name): action` — e.g. `TODO(jaeyong): verify the v2 benchmark numbers.`
  - `FIXME(name): issue` — e.g. `FIXME(gonghun): fix memory leak in the retry loop.`

## 3. Repository structure

A configuration-driven, modular layout. This repo uses a **flat layout** — the
importable package (`basicdet/`) sits at the **repo root**, not under `src/`
(mirroring BasicSR; the portable `src/`-layout default lives in the reusable
template at `~/.claude/templates/python-CLAUDE.md`). Keep the separation of
`configs / <package> / tests`.

```text
basicdet/                  # importable package AT THE REPO ROOT (no src/)
├── train.py  evaluate.py  predict.py   # entry points (python basicdet/train.py ...)
├── models/                # per-model-family pipelines
├── metrics/  utils/       # shared source modules
configs/                   # YAML experiment configs — no hardcoded values in Python
assets/data/               # datasets (gitignored)
runs/                      # training/eval outputs (gitignored)
tests/                     # pytest, mirrors the package layout
pyproject.toml             # tooling config + pinned dependencies
```

See **Project-specific guidance** below for the full annotated tree.

- Notebooks (if any) are for exploration only — production logic goes in the package.

## 4. Testing

- Use **`pytest`**. Tests live in `tests/` mirroring the source layout.
- Test the deterministic, logic-heavy pieces. Use small fixtures; keep tests fast
  and free of external dependencies (network, GPU, full datasets).
- Write a regression test when you fix a bug.

## 5. Git & workflow

- Small, focused commits with clear messages (imperative mood).
- Keep `.gitignore` current; never commit secrets, `__pycache__`, `.venv`, or
  large artifacts.
- Run formatter, linter, and type checker before committing (ideally via
  pre-commit hooks).

---

## Project-specific guidance

> A reusable master copy of the general sections above lives at
> `~/.claude/templates/python-CLAUDE.md` — start new projects from there.

**Purpose:** `basicdet` — a base framework for fine-tuning and evaluating
object-detection models (YOLO26, RF-DETR), applied here to person detection.
Data processing, model training, and detection evaluation only.

**Environment:** Managed with **conda** — env name `persondet`. Run
`conda activate persondet` before any work, then `pip install -e ".[dev]"`. Use
`python -m pip` inside the env (a bare `pip` can fall through to system pip). Run
GPU work on **GPU 1** (`CUDA_VISIBLE_DEVICES=1`); GPU 0 is reserved for the user.

### Repository structure (flat layout, BasicSR-style)

The package `basicdet/` lives at the repo root (not under `src/`), mirroring
BasicSR. It's organized **by model family**, because the network/loss/data are
owned by Ultralytics / RF-DETR — so there's no `archs/`/`losses/`/`data/` of our
own; the per-family module *is* the substance.

```text
basicdet/                       # flat package, pip-installed editable
├── train.py  evaluate.py  predict.py   # entry points: python basicdet/train.py --config ...
├── models/
│   ├── yolo.py                 # YOLO pipeline: train / evaluate / predict (Ultralytics)
│   └── rfdetr.py               # RF-DETR pipeline: train / evaluate / predict
├── metrics/coco.py             # COCO mAP, model-agnostic (used by RF-DETR eval)
└── utils/
    ├── config.py               # Pydantic schemas + load_experiment() (tagged union)
    ├── registry.py             # family -> pipeline dispatch (lazy per-family import)
    ├── tracking.py             # W&B integration
    └── seed.py  logging.py  runtime.py
configs/{yolo,rfdetr}/*.yaml    # one YAML per experiment; named <model><size>_person_<ver>
assets/data/persondet_v*/       # datasets (gitignored): YOLO images/+labels/ + COCO annotations/
runs/                           # outputs (gitignored): runs/detect/yolo26/<name>/ (Ultralytics
                                #   forces the runs/detect/ prefix), runs/rfdetr/<name>/
tests/                          # pytest — deterministic logic (config loading, etc.)
```

**Single config-driven entrypoint:** `basicdet/{train,evaluate,predict}.py` work
for both models (entry points inside the package, runnable as `python
basicdet/train.py` because it's pip-installed). The config's `family` field
(`yolo` | `rfdetr`) selects the pipeline via `basicdet/utils/registry.py`. Add a
model by adding a `models/<x>.py` (exposing `train`/`evaluate`/`predict`) and one
branch in the registry.

**Scope boundaries:** fine-tuning + evaluation of the **perception models feeding the
TRACE tracker**: person **detection** (YOLO26, RF-DETR) and person **ReID embedders**
(`reid_ftnet`, `reid_clipreid`, `reid_personvit` families; ReID checkpoints must stay
drop-in compatible with their TRACE-side loaders — `piapf/reid/` for ft_net/CLIP-ReID,
`apps/trace/worker/reid/personvit.py` for PersonViT — see the module docstrings). ReID intrinsic eval
(mAP/CMC on query–gallery) lives here; **tracking/MOT benchmarking** (BoT-SORT +
HOTA/MOTA/IDF1) is run downstream (`~/jordan/boxmot`, TRACE), **not here**. No
deployment code (TensorRT/ONNX serving export, inference-time quantization, serving
frameworks) — engine conversion happens in the TRACE repo. The official CLIP-ReID
trainer is vendored at `third_party/CLIP-ReID` (gitignored; clone command + pinned
commit in `basicdet/models/reid_clipreid.py`).

### ML-specific conventions

- **Tensor shapes & dtypes:** document them in docstrings/comments
  (e.g. `x: [batch_size, 3, H, W], torch.float16`). Use typed signatures.
- **Reproducibility:** `set_seed(seed)` seeds `random`, `numpy`, `torch`
  (+ cuDNN deterministic). Call at every entrypoint and log the seed.
- **Experiment tracking:** **W&B** (project `person-det`). Log metrics, full
  config, sample predictions. W&B auto-captures git commit. Never commit the API key.
- **Device handling:** never hardcode `.cuda()`; take a `device` arg (`"auto"`)
  and resolve in one place (`utils/runtime.py`).
- **No data/checkpoints in git** — `assets/data/` and `runs/` are gitignored.

### Working with a new dataset version (recurring gotchas — these have bitten us)

1. **Fix the stale path:** each `persondet_v*/data.yaml` ships with a macOS
   `path:` (`/Users/...`). Repoint it to the local absolute path before training.
2. **Verify labels:** confirm `images/<split>` count == `labels/<split>/*.txt`
   count for train/val/test. A partial copy = silent **empty-label training →
   mAP ≈ 0** (this happened on v1.1).
3. **Clear stale caches:** delete `labels/*.cache` so Ultralytics re-scans.
4. **RF-DETR layout:** RF-DETR needs `rfdetr/{train,valid,test}/` (images +
   `_annotations.coco.json`; note `valid`, not `val`). If a dataset only ships
   central `annotations/` + `images/`, build it via **hardlinks** (zero extra disk).
5. **Caching:** `cache: ram` for small/medium sets; `cache: disk` for very large
   (≳50k images, e.g. v4.2's 105k) — RAM caching would exhaust system memory.

### Long runs & resuming

- Run multi-hour trainings/evals in **tmux** (they must survive disconnects).
- They're resumable from the latest checkpoint: YOLO `resume=True`; RF-DETR via
  `resume:` in the config's `extra` → PyTorch-Lightning `ckpt_path` (restores
  epoch/optimizer/EMA). Per-epoch checkpoints make every run recoverable.

### Dataset provenance & leakage (important)

Datasets are versioned: **v1.1** (4 general sources), **v2.1** (CrowdHuman +
**MOT20**), **v3.1** (indoor CCTV), **v4.x** (large combined). Because **v2.1
contains MOT20**, never evaluate v2 models on MOT20 (train/test leakage). MOT17 is
leakage-free for v2 (different MOTChallenge sequences) but is same-family domain,
so strong MOT17 results partly reflect domain familiarity, not pure generalization.
