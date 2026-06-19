# CLAUDE.md

Guidance for writing and reviewing Python in this repository. The goal is code
that is **simple, readable, modular, and reproducible**.

> **Reusing this file:** Everything down to "Project-specific guidance" is
> general-purpose and applies to any Python project. Fill in or delete the
> project-specific section at the bottom per repo.

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

A configuration-driven, modular layout. Adapt names to the project; keep the
separation of `configs / src / scripts / tests`.

```text
project/
├── configs/                  # No hardcoded values in Python; everything lives here
├── src/                      # Core modular source code (importable package)
│   └── <package>/
│       ├── __init__.py
│       └── utils/
├── scripts/                  # Executable entry points (CLI orchestration)
├── tests/                    # Mirrors src/ layout
├── pyproject.toml            # Tooling config + pinned dependencies
└── README.md
```

- Notebooks (if any) are for exploration only — production logic goes in `src/`.

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

**Purpose:** Fine-tuning and evaluating object-detection models (YOLO26, RF-DETR)
for person detection — data processing, model training, and benchmarking only.

**Environment:** Managed with **conda** — env name `persondet`. Run
`conda activate persondet` before any work, then `pip install -e ".[dev]"`.

**Scope boundaries:** fine-tuning and evaluation only — no deployment code
(TensorRT/ONNX serving export, inference-time quantization, serving frameworks).
Deployment lives in a separate downstream repository.

### ML-specific conventions

- **Tensor shapes & dtypes:** document them in docstrings/comments
  (e.g. `x: [batch_size, 3, H, W], torch.float16`). Shape mismatches are the most
  common ML bug. Use `def forward(self, x: torch.Tensor) -> torch.Tensor:`, never
  untyped signatures.
- **Reproducibility:** a single `set_seed(seed)` seeds `random`, `numpy`, and
  `torch` (+ `cudnn.deterministic` when feasible). Call it at the top of every
  entrypoint and log the seed.
- **Experiment tracking:** use **Weights & Biases (W&B)** (project `person-det`).
  Log metrics, the full config, and sample predictions (images with predicted
  boxes). Every run traces to its exact config and git commit (W&B captures both
  automatically). Never commit the W&B API key.
- **No data in git.** Datasets and checkpoints go to DVC / cloud storage. Track
  data *versions*, not data. Persist train/val/test splits deterministically.
- **Device handling:** never hardcode `.cuda()`; take a `device` argument and
  auto-detect in one place.
- **Suggested layout:** `configs/model/{yolo26,rf_detr}.yaml`, `configs/data/`,
  `configs/train_config.yaml`; `src/<pkg>/{data,models,training,utils}/`;
  `scripts/{train,evaluate}.py`.
