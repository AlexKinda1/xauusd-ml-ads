# ADR 0001 — Initial project structure and tooling

- **Status** : Accepted
- **Date** : 2026-05-17
- **Phase** : 0 — Initialisation

## Context

We start a new ML/DL project for XAU/USD prediction (ADS, CESI). The repo must
be reproducible, scientifically rigorous, and friendly to both local
development and GPU training on Colab Pro / Kaggle.

## Decisions

### 1. Dependency management : Poetry
- Native `pyproject.toml` + lock file.
- Clear separation between runtime deps (`[tool.poetry.dependencies]`) and
  dev deps (`[tool.poetry.group.dev.dependencies]`).
- Lock file committed for reproducibility.

### 2. Python 3.11+
- Required by recent libraries (mlflow 2.16, transformers 4.44).
- Upper bound `<3.13` to avoid bleeding-edge incompatibilities with PyTorch
  and pandas-ta at project start.

### 3. NumPy pinned `<2.0`
- pandas-ta and several TS / TA libs were not yet fully NumPy 2.x compatible
  at the start of this project. Re-evaluate at Phase 4.

### 4. PyTorch CPU by default
- Local dev (data collection, feature engineering) does not need a GPU.
- For training on Colab/Kaggle, torch is already pre-installed with CUDA.
- A note in `pyproject.toml` documents how to override locally with a CUDA
  wheel.

### 5. Config in YAML, NOT Hydra
- Hydra adds runtime complexity (composition, overrides via CLI) we do not
  yet need. Plain YAML loaded via `src/utils/config.py` is enough for now.
- If multi-run sweeps become painful, we can migrate to Hydra later without
  pain.

### 6. Source layout : `src/` package
- Importable as `from src.utils.seed import set_global_seed`.
- Avoids common path-manipulation hacks.
- Listed in `tool.poetry.packages` so `poetry install` puts it on `sys.path`.

### 7. Data folder structure : raw / interim / processed / external
- Standard Cookiecutter Data Science convention.
- All four folders are gitignored except `.gitkeep` to preserve the tree.

### 8. MLflow as the single tracking system
- Local file-based store (`./mlruns/`) by default; works offline.
- `MLFLOW_TRACKING_URI` can be overridden via `.env` for a remote server.

### 9. Anti-leakage tests are first-class
- A dedicated `tests/test_anti_leakage.py` will be written BEFORE any feature
  pipeline. Marked with `pytest -m leakage` for fast targeted runs.

## Consequences

- Onboarding cost is one-time : `make setup` is the single command.
- Strict mypy mode on `src/` enforces type discipline early — slows down
  initial coding but pays off when refactoring across 7 model implementations.
- Notebook outputs are NOT versioned (`.gitignore` blocks `.ipynb_checkpoints`).
  Final figures go to `reports/figures/` via scripts, not via notebook saves.
