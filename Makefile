.PHONY: all setup preprocess features train evaluate shap dashboard verify clean

PYTHON := python

# ── One-shot pipeline ─────────────────────────────────────────────
all: preprocess features train evaluate shap

# ── Environment ───────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Data pipeline ─────────────────────────────────────────────────
preprocess:
	$(PYTHON) scripts/preprocess.py

features:
	$(PYTHON) scripts/feature_engineering.py

# ── Modeling ──────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/train_models.py

# tune target requires scripts/hyperparameter_tuning.py (not yet implemented)
# tune:
# 	$(PYTHON) scripts/hyperparameter_tuning.py

evaluate:
	$(PYTHON) scripts/evaluate.py

# ── Interpretability ──────────────────────────────────────────────
shap:
	$(PYTHON) scripts/shap_analysis.py

# ── Dashboard ─────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Quality gates ─────────────────────────────────────────────────
lint:
	ruff check scripts/ tests/ --ignore E501,E402

test:
	pytest tests/ -v --tb=short

sql-lint:
	@echo "No SQL in this project — skipping sqlfluff"

docker-build:
	@echo "No Dockerfile in this project — skipping docker-build"

verify: lint format-check test audit
	@echo "All quality gates passed"

# ── Utilities ─────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# === Quality gates (extended) ===

format:
	ruff format scripts/ dashboard/

format-check:
	ruff format --check scripts/ dashboard/

audit:
	$(PYTHON) scripts/audit_consistency.py
