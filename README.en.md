<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0-green?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0-blue?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.42-orange?logo=shap&logoColor=white" alt="SHAP">
  <a href="https://github.com/MeaFew/riskscore/actions"><img src="https://github.com/MeaFew/riskscore/workflows/CI/badge.svg" alt="CI"></a>
</p>

## Overview

End-to-end credit risk scoring pipeline built on the Kaggle Home Credit Default Risk dataset. Implements professional-grade feature engineering (WOE/IV), model comparison (LR -> RF -> XGBoost -> LightGBM), and SHAP-based interpretability (SHAP).

## Key Highlights

- **Feature Engineering**: WOE binning (analytic reference), per-fold target encoding (leakage-free), cross-features
- **Model Stack**: Logistic Regression (baseline) -> Random Forest -> XGBoost -> LightGBM
- **Evaluation**: AUC, KS, Gini, calibration, confusion matrix at optimal threshold
- **Interpretability**: SHAP summary, dependence plots, force plot for individual cases
- **Delivery**: Streamlit dashboard with risk calculator

## Tech Stack

| Layer | Tools | Notes |
|-------|-------|-------|
| ETL | pandas, scikit-learn | Missing value imputation, outlier capping |
| Feature Eng | Custom WOE/IV | Quantile-based binning with smoothing |
| Modeling | XGBoost, LightGBM, sklearn | 5-fold stratified CV |
| Interpretability | SHAP | TreeExplainer for gradient boosting models |
| Evaluation | scipy, sklearn | AUC, KS, Gini, PR curve, calibration |
| Delivery | Streamlit | Interactive risk calculator + model comparison |
| Quality | pytest, ruff, GitHub Actions | CI runs lint + tests on every push |

## Quick Start

```bash
git clone https://github.com/MeaFew/riskscore.git
cd riskscore

# Download real dataset (GitHub Releases, ~40MB)
bash download_data.sh

# Run full pipeline
make all

# Or step by step
make preprocess
make features
make train
make evaluate
make shap

# Launch dashboard
make dashboard

# Quality gates
make verify
```

## Project Structure

```
.
├── scripts/
│   ├── generate_mock_data.py     # Synthetic data generator (for CI)
│   ├── preprocess.py              # Data cleaning & missing value handling
│   ├── feature_engineering.py     # cross-features + WOE/IV analytic report (target encoding moved into CV Pipeline)
│   ├── train_models.py            # LR / RF / XGB / LGBM with CV
│   ├── evaluate.py                # ROC, PR, calibration, confusion matrix
│   └── shap_analysis.py           # SHAP summary, dependence, force plots
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
├── tests/
│   └── test_pipeline.py           # Unit + integration tests
├── config.py                      # Centralized paths & hyperparameters
├── Makefile                       # Workflow orchestration
└── requirements.txt
```

## Model Performance

### Benchmark

Based on [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (7,190+ teams, metric: AUC-ROC).

| Reference | AUC | Notes |
|-----------|-----|-------|
| Kaggle Starter Baseline | 0.688 | Official starter notebook, no feature engineering |
| Single-table Logistic Regression | 0.748 | `application_train` only + GridSearchCV |
| Single-table LightGBM | 0.749 | Same as above, gradient boosting |
| Competition Median | ~0.72-0.75 | Leaderboard median |
| Competition Top 10% | ~0.795 | Multi-table features + ensemble |
| **This Project (single-table)** | **0.766** | Per-fold target encoding (leakage-free) + XGBoost/LightGBM (5-fold CV / OOF) |
| This Project (multi-table, planned) | ~0.78 (projected) | Requires full auxiliary tables; this repo implements single-table only |

> Note: Competition Private Leaderboard is closed. Scores above are from local 5-fold stratified cross-validation plus leakage-free out-of-fold (OOF) evaluation on real Kaggle data (307,511 train samples, `application_train`). Multi-table results require running `scripts/aggregate_auxiliary_features.py` and `scripts/merge_auxiliary_features.py` before `make features`.

### Results

| Model | AUC | KS | Gini |
|-------|-----|-----|------|
| Logistic Regression | 0.626 | 0.192 | 0.251 |
| Random Forest | 0.746 | 0.365 | 0.491 |
| XGBoost | **0.766** | **0.399** | **0.533** |
| LightGBM | **0.766** | **0.398** | **0.532** |

> Values from 5-fold stratified cross-validation on real Kaggle data (307,511 samples) with single-table features (application_train only). Target encoding is fit **per CV fold inside a sklearn Pipeline** (validation rows never encode their own target — no leakage). XGBoost OOF AUC = **0.766** (each row scored by a model that never saw it — a faithful generalization estimate).

### Leakage fixes (important)

This pipeline corrects two common credit-scoring leakage patterns:

- **Target-encoding leakage**: an earlier version computed target encoding on the full training set, then fed it into 5-fold CV — so a validation row's own target leaked into its encoded feature and inflated AUC. Target encoding now lives in a sklearn `Pipeline` and is fit on each fold's training split only. IV-based feature selection was removed from the modeling path (kept only as the analytic `data/processed/iv_report.csv`) to avoid the same class of leak via "select features with the full-set target, then use them inside CV folds".
- **Fabricated test AUC**: Home Credit's `application_test.csv` has no `TARGET`, so there is no labeled holdout. An earlier version split the training set 80/20, retrained on the full data, and evaluated — producing a 0.80 "test AUC" that was a resubstitution number and, worse, higher than the honest CV AUC. This metric was removed; we now report OOF AUC.
- **Preprocessing leakage**: outlier caps and median imputation are now fit on train only, then transformed onto test.

## Related Projects

| Project | Repo | Description |
|---------|------|-------------|
| E-commerce User Analytics | [MeaFew/shoplytics](https://github.com/MeaFew/shoplytics) | 29M real user behavior records, 10 analytical modules |
| Marketing Attribution & MMM | [MeaFew/attributor](https://github.com/MeaFew/attributor) | MMM + multi-touch attribution + budget optimization |
| Multivariate Time Series | [MeaFew/foresight](https://github.com/MeaFew/foresight) | LSTM / Transformer / XGBoost time series forecasting |
| Graph Fraud Detection | [MeaFew/graphguard](https://github.com/MeaFew/graphguard) | GNN illicit transaction detection (Elliptic) |

## License

MIT
