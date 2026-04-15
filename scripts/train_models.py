"""Train and compare multiple credit risk models.

Models:
- Logistic Regression (baseline, highly interpretable)
- Random Forest (ensemble, non-linear)
- XGBoost (gradient boosting, industry standard)
- LightGBM (gradient boosting, faster training)

Output: trained models + cross-validation results JSON.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    ks_score,
)

warnings.filterwarnings("ignore")



def gini_score(y_true, y_proba):
    """Gini = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_proba) - 1


def cross_validate_model(model, X, y, cv=5):
    """Stratified K-Fold cross-validation with AUC, KS, Gini."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    aucs, kss, ginis = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]

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


def train_logistic_regression(X, y):
    """Train logistic regression with class weight balancing."""
    print("Training Logistic Regression ...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return model, cross_validate_model(model, X, y)


def train_random_forest(X, y):
    """Train random forest."""
    print("Training Random Forest ...")
    model = RandomForestClassifier(**RF_PARAMS)
    return model, cross_validate_model(model, X, y)


def train_xgboost(X, y):
    """Train XGBoost classifier."""
    print("Training XGBoost ...")
    model = xgb.XGBClassifier(
        **{k: v for k, v in XGB_PARAMS.items() if k not in ("early_stopping_rounds",)}
    )

    # Manual CV with early stopping
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs, kss, ginis = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_proba))
        kss.append(ks_score(y_val, y_proba))
        ginis.append(gini_score(y_val, y_proba))

    metrics = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ks_mean": float(np.mean(kss)),
        "ks_std": float(np.std(kss)),
        "gini_mean": float(np.mean(ginis)),
        "gini_std": float(np.std(ginis)),
    }
    return model, metrics


def train_lightgbm(X, y):
    """Train LightGBM classifier."""
    print("Training LightGBM ...")
    model = lgb.LGBMClassifier(
        **{k: v for k, v in LGB_PARAMS.items() if k not in ("early_stopping_rounds", "verbose")}
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs, kss, ginis = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        y_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_proba))
        kss.append(ks_score(y_val, y_proba))
        ginis.append(gini_score(y_val, y_proba))

    metrics = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ks_mean": float(np.mean(kss)),
        "ks_std": float(np.std(kss)),
        "gini_mean": float(np.mean(ginis)),
        "gini_std": float(np.std(ginis)),
    }
    return model, metrics


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

    print(f"Features: {X.shape[1]}, Samples: {len(X)}, Default rate: {y.mean()*100:.2f}%\n")

    # Train all models
    results = {}
    models = {}

    models["logistic_regression"], results["logistic_regression"] = train_logistic_regression(X, y)
    models["random_forest"], results["random_forest"] = train_random_forest(X, y)
    models["xgboost"], results["xgboost"] = train_xgboost(X, y)
    models["lightgbm"], results["lightgbm"] = train_lightgbm(X, y)

    # Save best model (highest AUC) — retrain on full data
    best_model_name = max(results, key=lambda k: results[k]["auc_mean"])
    print(f"\nBest model: {best_model_name} (CV AUC={results[best_model_name]['auc_mean']:.4f})")
    print(f"Retraining {best_model_name} on full training set ...")

    # Recreate model from scratch, then fit on full data
    if best_model_name == "logistic_regression":
        best_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")
        best_model.fit(X, y)
    elif best_model_name == "random_forest":
        best_model = RandomForestClassifier(**RF_PARAMS)
        best_model.fit(X, y)
    elif best_model_name == "xgboost":
        best_model = xgb.XGBClassifier(**{k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"})
        best_model.fit(X, y)
    elif best_model_name == "lightgbm":
        from sklearn.model_selection import train_test_split as _tts
        X_rt, X_val, y_rt, y_val = _tts(X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
        best_model = lgb.LGBMClassifier(**{k: v for k, v in LGB_PARAMS.items() if k not in ("early_stopping_rounds", "verbose")})
        best_model.fit(X_rt, y_rt, eval_set=[(X_val, y_val)])
    else:
        best_model = models[best_model_name]

    # Save model in both formats for maximum compatibility
    if hasattr(best_model, "save_model"):
        best_model.save_model(str(MODEL_PATH.with_suffix(".json")))
    joblib.dump(best_model, str(MODEL_PATH.with_suffix(".joblib")))

    # Save results
    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump({
            "cv_results": results,
            "best_model": best_model_name,
            "feature_count": X.shape[1],
            "sample_count": len(X),
            "default_rate": float(y.mean()),
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved results to {MODEL_RESULTS_JSON}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'AUC':>10} {'KS':>10} {'Gini':>10}")
    print("=" * 70)
    for name, metrics in results.items():
        marker = " *" if name == best_model_name else ""
        print(f"{name:<20} {metrics['auc_mean']:>10.4f} {metrics['ks_mean']:>10.4f} {metrics['gini_mean']:>10.4f}{marker}")
    print("=" * 70)


if __name__ == "__main__":
    main()
