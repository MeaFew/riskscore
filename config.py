"""Project-wide configuration for credit-risk-scoring.

All paths are resolved relative to this file's location.
"""

from pathlib import Path

# ── Base directories ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
IMAGES_DIR = Path(__file__).resolve().parent / "images"

# ── Input files ───────────────────────────────────────────────────
TRAIN_CSV = RAW_DATA_DIR / "application_train.csv"
TEST_CSV = RAW_DATA_DIR / "application_test.csv"

# ── Output files ──────────────────────────────────────────────────
CLEANED_TRAIN_CSV = PROCESSED_DATA_DIR / "application_train_cleaned.csv"
CLEANED_TEST_CSV = PROCESSED_DATA_DIR / "application_test_cleaned.csv"
FEATURES_TRAIN_CSV = PROCESSED_DATA_DIR / "features_train.csv"
FEATURES_TEST_CSV = PROCESSED_DATA_DIR / "features_test.csv"
MODEL_PATH = MODELS_DIR / "best_risk_model"  # stem (no extension); use with .json or .joblib
MODEL_RESULTS_JSON = REPORTS_DIR / "model_results.json"
SHAP_SUMMARY_PNG = IMAGES_DIR / "shap_summary.png"


# ── Shared utility functions ─────────────────────────────────────
import scipy.stats as _scipy_stats

def ks_score(y_true, y_proba):
    """Compute Kolmogorov-Smirnov statistic."""
    pos = y_proba[y_true == 1]
    neg = y_proba[y_true == 0]
    return _scipy_stats.ks_2samp(pos, neg).statistic

# ── Modeling constants ────────────────────────────────────────────
TARGET_COL = "TARGET"
RANDOM_STATE = 42

# ── Feature engineering ───────────────────────────────────────────
# WOE binning
WOE_MIN_SAMPLES = 0.05  # (reserved for future use) minimum bin size as fraction of total
WOE_MAX_BINS = 10

# IV threshold for feature selection
IV_THRESHOLD = 0.02  # features with IV < 0.02 are considered weak

# PSI threshold for stability monitoring

# ── Model hyperparameters (tuned via Optuna in practice) ──────────
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,  # handle class imbalance
    "random_state": RANDOM_STATE,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "tree_method": "hist",
}

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,
    "random_state": RANDOM_STATE,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 50,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
