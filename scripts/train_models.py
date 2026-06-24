"""Train and compare multiple credit risk models.

Models:
- Logistic Regression (baseline, highly interpretable)
- Random Forest (ensemble, non-linear)
- XGBoost (gradient boosting, industry standard)
- LightGBM (gradient boosting, faster training)

Each model is wrapped in a sklearn ``Pipeline`` of the form
``TargetEncoder -> classifier``. The target encoder transforms high-cardinality
categorical columns, and because it lives inside the pipeline it is fit on each
CV fold's training split only — so validation rows never encode their own
target (leakage fix). The persisted artifact is the full pipeline, so SHAP /
evaluation / dashboard all consume a leakage-free, self-contained model.

Output: trained models + cross-validation results JSON.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_utils import ks_score

from config import (
    FEATURES_TRAIN_CSV,
    LGB_PARAMS,
    MODEL_PATH,
    MODEL_RESULTS_JSON,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RF_PARAMS,
    TARGET_COL,
    XGB_PARAMS,
)

# Silence only the noisy LightGBM/XGBoost fitting chatter, not genuine
# convergence / deprecation warnings from sklearn (a blanket "ignore"
# previously hid real signal such as LogisticRegression not converging).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*lightgbm.*", category=Warning)


def gini_score(y_true, y_proba):
    """Gini = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_proba) - 1


def _split_cat_numeric(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify high-cardinality categoricals (target-encoded) vs everything else.

    Low-cardinality categoricals were already one-hot encoded upstream in
    feature_engineering, so any remaining string/object columns are the high-card
    TE candidates. Numeric columns pass through untouched. Both 'object' (legacy)
    and pandas 'string' dtypes are treated as categorical so this is robust
    across pandas 2 (object) and pandas 3/4 (string).
    """
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, numeric_cols


def make_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    """Build a fresh ``TargetEncoder -> classifier`` pipeline for one model.

    A new pipeline (with fresh, unfitted steps) is created for each CV fold via
    this factory, so no fold inherits a partially-fit encoder from another.
    """
    cat_cols, numeric_cols = _split_cat_numeric(X)
    preprocessor = ColumnTransformer(
        transformers=[
            ("te", TargetEncoder(target_type="binary", random_state=RANDOM_STATE), cat_cols),
        ],
        remainder="passthrough",  # numeric cols pass through untouched
        verbose_feature_names_out=False,
    )

    if model_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs"
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(**RF_PARAMS)
    elif model_name == "xgboost":
        clf = xgb.XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items() if k not in ("early_stopping_rounds",)}
        )
    elif model_name == "lightgbm":
        clf = lgb.LGBMClassifier(
            **{k: v for k, v in LGB_PARAMS.items() if k not in ("early_stopping_rounds", "verbose")}
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def _eval_set_fit(pipeline: Pipeline, X_train, y_train, X_val, y_val):
    """Fit a pipeline while passing the val split as eval_set for tree early stopping.

    sklearn pipelines route fit kwargs only to the LAST step, so we fit the
    preprocessing on train, transform both splits, then fit the estimator with
    ``eval_set``. This mirrors the previous early-stopping behavior while
    keeping the encoder leakage-free (fit on train split only).
    """
    # Only XGBoost/LightGBM reach here. Fit preprocess on train split only.
    Xt = pipeline.named_steps["preprocess"].fit_transform(X_train, y_train)
    Xv = pipeline.named_steps["preprocess"].transform(X_val)
    pipeline.named_steps["model"].fit(Xt, y_train, eval_set=[(Xv, y_val)])


def cross_validate_model(model_name: str, X, y, cv=5):
    """Stratified K-Fold cross-validation with a fresh leakage-free pipeline per fold.

    For XGBoost/LightGBM the validation split is supplied as ``eval_set`` for
    early stopping (mirroring the original behavior), with preprocessing fit
    on the training split only.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    use_early_stop = model_name in ("xgboost", "lightgbm")

    aucs, kss, ginis = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = make_pipeline(model_name, X)
        if use_early_stop:
            _eval_set_fit(pipeline, X_train, y_train, X_val, y_val)
        else:
            pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        aucs.append(roc_auc_score(y_val, y_proba))
        kss.append(ks_score(y_val, y_proba))
        ginis.append(gini_score(y_val, y_proba))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ks_mean": float(np.mean(kss)),
        "ks_std": float(np.std(kss)),
        "gini_mean": float(np.mean(ginis)),
        "gini_std": float(np.std(ginis)),
    }


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def train_all(X, y):
    """Cross-validate every model and return results dict."""
    results = {}
    for name in MODEL_NAMES:
        print(f"Training {name} ...")
        results[name] = cross_validate_model(name, X, y)
    return results


def retrain_best(model_name: str, X, y):
    """Retrain the best model's pipeline on the FULL training set for deployment."""
    pipeline = make_pipeline(model_name, X)
    if model_name == "xgboost":
        from sklearn.model_selection import train_test_split as _tts

        X_rt, X_val, y_rt, y_val = _tts(X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
        _eval_set_fit(pipeline, X_rt, y_rt, X_val, y_val)
    elif model_name == "lightgbm":
        from sklearn.model_selection import train_test_split as _tts

        X_rt, X_val, y_rt, y_val = _tts(X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
        _eval_set_fit(pipeline, X_rt, y_rt, X_val, y_val)
    else:
        pipeline.fit(X, y)
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=FEATURES_TRAIN_CSV)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.train} ...")
    df = pd.read_csv(args.train)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL, "SK_ID_CURR"])

    print(f"Features: {X.shape[1]}, Samples: {len(X)}, Default rate: {y.mean() * 100:.2f}%\n")

    # Cross-validate every model (leakage-free: encoder fit per fold)
    results = train_all(X, y)

    # Save best model (highest AUC) — retrain pipeline on full data
    best_model_name = max(results, key=lambda k: results[k]["auc_mean"])
    print(f"\nBest model: {best_model_name} (CV AUC={results[best_model_name]['auc_mean']:.4f})")
    print(f"Retraining {best_model_name} pipeline on full training set ...")
    best_pipeline = retrain_best(best_model_name, X, y)

    # Persist the full pipeline (preprocess + model) as joblib so downstream
    # consumers (SHAP / evaluate / dashboard) reload a self-contained,
    # leakage-free model. The native XGBoost .json no longer carries the
    # fitted encoder, so joblib is now the single source of truth.
    joblib.dump(best_pipeline, str(MODEL_PATH.with_suffix(".joblib")))
    # Drop any stale .json from the previous (non-pipeline) run to avoid
    # find_best_model_path loading an encoder-less artifact.
    stale_json = MODEL_PATH.with_suffix(".json")
    if stale_json.exists():
        stale_json.unlink()
        print(f"  Removed stale (encoder-less) checkpoint: {stale_json}")

    # Save results
    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump(
            {
                "cv_results": results,
                "best_model": best_model_name,
                "feature_count": X.shape[1],
                "sample_count": len(X),
                "default_rate": float(y.mean()),
                "leakage_notes": {
                    "target_encoding": "fit per CV fold via sklearn Pipeline; validation rows never encode their own target",
                    "test_metrics": "no labeled test set available (Home Credit application_test.csv has no TARGET); report 5-fold CV AUC only",
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved model to {MODEL_PATH}.joblib")
    print(f"Saved results to {MODEL_RESULTS_JSON}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'AUC':>10} {'KS':>10} {'Gini':>10}")
    print("=" * 70)
    for name, metrics in results.items():
        marker = " *" if name == best_model_name else ""
        print(
            f"{name:<20} {metrics['auc_mean']:>10.4f} {metrics['ks_mean']:>10.4f} {metrics['gini_mean']:>10.4f}{marker}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
