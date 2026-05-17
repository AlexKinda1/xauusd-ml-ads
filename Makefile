# XAU/USD ML/DL — project Makefile
# Run `make help` to list available commands.

.PHONY: help setup install lock update lint format type test test-leakage clean \
        collect features train-all evaluate report

PYTHON := python
POETRY := poetry

help:  ## Display this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------- Environment ----------
setup: install  ## Full setup: install deps + pre-commit + .env
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Edit .env with your API keys."

install:  ## Install Poetry dependencies
	$(POETRY) install --with dev

lock:  ## Refresh poetry.lock without installing
	$(POETRY) lock --no-update

update:  ## Update dependencies within version constraints
	$(POETRY) update

# ---------- Quality ----------
lint:  ## Run ruff linter
	$(POETRY) run ruff check src tests scripts

format:  ## Auto-format with ruff + black
	$(POETRY) run ruff check --fix src tests scripts
	$(POETRY) run black src tests scripts

type:  ## Strict type checking with mypy
	$(POETRY) run mypy src

test:  ## Run all tests with coverage
	$(POETRY) run pytest

test-leakage:  ## Run only anti-leakage tests
	$(POETRY) run pytest -m leakage -v

# ---------- Pipeline (placeholders, implemented in later phases) ----------
collect:  ## Phase 1: collect OHLCV + macro + sentiment data
	$(POETRY) run python scripts/01_collect_all_data.py

features:  ## Phase 2: build features + targets
	$(POETRY) run python scripts/02_build_features.py

train-all:  ## Phase 4: train all 7 models on both tasks
	$(POETRY) run python scripts/03_build_splits.py
	$(POETRY) run python scripts/04a_train_baseline.py
	$(POETRY) run python scripts/04b_train_xgboost.py

mlflow-ui:  ## Open MLflow tracking UI on http://localhost:5000
	$(POETRY) run mlflow ui --backend-store-uri ./mlruns

evaluate:  ## Phase 5: evaluate and compare all models
	$(POETRY) run python scripts/04_evaluate_all.py

report:  ## Phase 5: generate final figures and tables
	$(POETRY) run python scripts/05_generate_report.py

# ---------- Cleanup ----------
clean:  ## Remove caches and build artefacts (preserves data/ and mlruns/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
