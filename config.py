"""Project-wide configuration for credit-risk-scoring.

All paths are resolved relative to this file's location.
"""

from pathlib import Path

# ── Base directories ──────────────────────────────────────────────
# BASE_DIR = Path(__file__).resolve().parent  # (unused, reserved for future)
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# EXTERNAL_DATA_DIR = DATA_DIR / "external"  # (unused, reserved for future)
MODELS_DIR = Path(__file__).resolve().parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
IMAGES_DIR = Path(__file__).resolve().parent / "images"
# DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"  # (unused, reserved for future)
# DOCS_DIR = Path(__file__).resolve().parent / "docs"  # (unused, reserved for future)

# ── Input files ───────────────────────────────────────────────────
TRAIN_CSV = RAW_DATA_DIR / "application_train.csv"
TEST_CSV = RAW_DATA_DIR / "application_test.csv"
# BUREAU_CSV = RAW_DATA_DIR / "bureau.csv"  # (unused, reserved for future)
# BUREAU_BALANCE_CSV = RAW_DATA_DIR / "bureau_balance.csv"  # (unused, reserved for future)
# PREV_APP_CSV = RAW_DATA_DIR / "previous_application.csv"  # (unused, reserved for future)
# POS_CASH_CSV = RAW_DATA_DIR / "POS_CASH_balance.csv"  # (unused, reserved for future)
# CC_BALANCE_CSV = RAW_DATA_DIR / "credit_card_balance.csv"  # (unused, reserved for future)
# INSTALLMENTS_CSV = RAW_DATA_DIR / "installments_payments.csv"  # (unused, reserved for future)

# ── Output files ──────────────────────────────────────────────────
CLEANED_TRAIN_CSV = PROCESSED_DATA_DIR / "application_train_cleaned.csv"
CLEANED_TEST_CSV = PROCESSED_DATA_DIR / "application_test_cleaned.csv"
FEATURES_TRAIN_CSV = PROCESSED_DATA_DIR / "features_train.csv"
FEATURES_TEST_CSV = PROCESSED_DATA_DIR / "features_test.csv"
MODEL_PATH = MODELS_DIR / "xgb_risk_model"  # stem (no extension); use with .json or .joblib
MODEL_RESULTS_JSON = REPORTS_DIR / "model_results.json"
SHAP_SUMMARY_PNG = IMAGES_DIR / "shap_summary.png"
# ROC_CURVE_PNG = IMAGES_DIR / "roc_curve.png"  # (unused, reserved for future)
# FEATURE_IMPORTANCE_PNG = IMAGES_DIR / "feature_importance.png"  # (unused, reserved for future)

# ── Modeling constants ────────────────────────────────────────────
TARGET_COL = "TARGET"
RANDOM_STATE = 42
# TEST_SIZE = 0.2  # (unused, reserved for future)

# ── Feature engineering ───────────────────────────────────────────
# WOE binning
WOE_MIN_SAMPLES = 0.05  # minimum bin size as fraction of total
WOE_MAX_BINS = 10

# IV threshold for feature selection
IV_THRESHOLD = 0.02  # features with IV < 0.02 are considered weak

# PSI threshold for stability monitoring
# PSI_THRESHOLD = 0.25  # >0.25 indicates significant distribution shift  # (unused, reserved for future)

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
