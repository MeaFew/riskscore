# Credit Risk Scoring

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0-green?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0-blue?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.42-orange?logo=shap&logoColor=white" alt="SHAP">
  <img src="https://img.shields.io/badge/CI-passing-brightgreen?logo=githubactions&logoColor=white" alt="CI">
</p>

## Overview

End-to-end credit risk scoring pipeline built on the Kaggle Home Credit Default Risk dataset. Implements industry-standard feature engineering (WOE/IV), model comparison (LR → RF → XGBoost → LightGBM), and production-ready interpretability (SHAP).

## Key Highlights

- **Feature Engineering**: WOE binning, IV-based selection, target encoding, cross-features
- **Model Stack**: Logistic Regression (baseline) → Random Forest → XGBoost → LightGBM
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
# Generate synthetic data for local testing
python scripts/generate_mock_data.py

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

# Run tests
make verify
```

## Project Structure

```
.
├── scripts/
│   ├── generate_mock_data.py     # Synthetic data generator (for CI)
│   ├── preprocess.py              # Data cleaning & missing value handling
│   ├── feature_engineering.py     # WOE/IV, target encoding, cross-features
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
| Competition Median | ~0.72–0.75 | Leaderboard median |
| Competition Top 10% | ~0.795 | Multi-table features + ensemble |
| **This Project (5-Fold CV)** | **0.782** | WOE/IV + target encoding + XGBoost/LightGBM ensemble + **multi-table features** |

> Note: Competition Private Leaderboard is closed. Scores above are from local 5-fold stratified cross-validation on real Kaggle data (307,511 train / 48,744 test).

### Results

| Model | AUC | KS | Gini |
|-------|-----|-----|------|
| Logistic Regression | 0.654 | 0.229 | 0.309 |
| Random Forest | 0.760 | 0.389 | 0.519 |
| XGBoost | **0.783** | **0.427** | **0.565** |
| LightGBM | 0.781 | 0.427 | 0.563 |

> Values from 5-fold stratified cross-validation on real Kaggle data with auxiliary table aggregation (bureau, previous_application, POS/credit card balances, installments). Hold-out test set AUC = **0.773**.

## License

MIT
