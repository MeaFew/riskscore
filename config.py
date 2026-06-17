"""Project-wide configuration for riskscore.

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
MODEL_PATH = MODELS_DIR / "xgb_risk_model"  # stem (no extension); use with .json or .joblib
MODEL_RESULTS_JSON = REPORTS_DIR / "model_results.json"
SHAP_SUMMARY_PNG = IMAGES_DIR / "shap_summary.png"


# ── Modeling constants ────────────────────────────────────────────
# NOTE: metric helpers (e.g. ks_score) live in scripts/metrics_utils.py so that
# importing this config module does not require scipy.
TARGET_COL = "TARGET"
RANDOM_STATE = 42

# ── Feature engineering ───────────────────────────────────────────
# WOE binning
WOE_MAX_BINS = 10

# IV threshold for feature selection
IV_THRESHOLD = 0.02  # features with IV < 0.02 are considered weak

# PSI threshold for stability monitoring
PSI_THRESHOLD = 0.1  # Population Stability Index threshold for monitoring

# ── Model hyperparameters (hand-tuned defaults; tune per dataset) ─
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
