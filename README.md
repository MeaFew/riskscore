# Credit Risk Scoring

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0-green?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0-blue?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.42-orange" alt="SHAP">
  <a href="https://github.com/MeaFew/credit-risk-scoring/actions"><img src="https://github.com/MeaFew/credit-risk-scoring/workflows/CI/badge.svg" alt="CI"></a>
</p>

<p align="center">
  🏠 <b>主仓：<a href="https://gitee.com/zeroonei1/credit-risk-scoring">Gitee</a></b> &nbsp;|&nbsp;
  🔗 <a href="https://github.com/MeaFew/credit-risk-scoring">GitHub（自动同步）</a>
</p>

<p align="center">
  <b>中文</b> | <a href="./README.en.md">English</a>
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
# 从 Gitee 克隆（国内推荐，速度更快）
git clone https://gitee.com/zeroonei1/credit-risk-scoring.git

# 或从 GitHub
git clone https://github.com/MeaFew/credit-risk-scoring.git
cd credit-risk-scoring

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
| **This Project (single-table)** | **0.763** | WOE/IV + target encoding + XGBoost/LightGBM (5-fold CV) |
| **This Project (multi-table)** | **0.783** | (expected, requires full Kaggle data with auxiliary tables) |

> Note: Competition Private Leaderboard is closed. Scores above are from local 5-fold stratified cross-validation on real Kaggle data (307,511 train / 48,744 test). Multi-table results require running `scripts/aggregate_auxiliary_features.py` and `scripts/merge_auxiliary_features.py` before `make features`.

### Results

| Model | AUC | KS | Gini |
|-------|-----|-----|------|
| Logistic Regression | 0.634 | 0.205 | 0.268 |
| Random Forest | 0.745 | 0.366 | 0.490 |
| XGBoost | **0.762** | **0.394** | **0.525** |
| LightGBM | **0.763** | **0.394** | **0.526** |

> Values from 5-fold stratified cross-validation on real Kaggle data with single-table features (application_train only). Hold-out test set AUC = **0.766** (LightGBM). Multi-table features (bureau, previous_application, etc.) can push AUC to **0.783** by running the auxiliary aggregation scripts.

## 相关项目

| 项目 | Gitee（主仓） | GitHub（镜像） |
|------|---------------|-----------------|
| 电商用户行为分析 | [Gitee](https://gitee.com/zeroonei1/ecommerce-user-analytics) | [GitHub](https://github.com/MeaFew/ecommerce-user-analytics) |
| 营销归因与预算优化 | [Gitee](https://gitee.com/zeroonei1/marketing-attribution-mmm) | [GitHub](https://github.com/MeaFew/marketing-attribution-mmm) |
| 多元时序预测 | [Gitee](https://gitee.com/zeroonei1/multivariate-timeseries-forecasting) | [GitHub](https://github.com/MeaFew/multivariate-timeseries-forecasting) |

## License

MIT
