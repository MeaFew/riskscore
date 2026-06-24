"""Model evaluation and calibration for riskscore.

Generates (using leakage-free OUT-OF-FOLD predictions on the training set):
- ROC curve plot
- Precision-Recall curve plot
- Calibration plot (reliability diagram)
- Score distribution plot
- Confusion matrix at optimal threshold

The Home Credit ``application_test.csv`` ships WITHOUT a TARGET column, so
there is no labeled holdout to evaluate against. Rather than fabricate a "test
AUC" by retraining on a training subset and scoring it (a resubstitution /
leakage artifact), this module reports metrics on OUT-OF-FOLD (OOF) predictions
produced by the same leakage-free per-fold pipeline used in training: each row's
score comes from a model that did NOT see that row during fitting. OOF metrics
are a faithful estimate of generalization and are comparable to the reported
5-fold CV AUC.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_utils import ks_score

from config import (
    FEATURES_TRAIN_CSV,
    IMAGES_DIR,
    MODEL_PATH,
    MODEL_RESULTS_JSON,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
)


def plot_roc_curve(y_true, y_proba, save_path: Path):
    """Plot ROC curve and save."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Out-of-Fold)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve: {save_path}")


def plot_pr_curve(y_true, y_proba, save_path: Path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Out-of-Fold)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved PR curve: {save_path}")


def plot_calibration(y_true, y_proba, save_path: Path):
    """Plot calibration (reliability) diagram."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Plot (Out-of-Fold, Reliability Diagram)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved calibration plot: {save_path}")


def plot_score_distribution(y_true, y_proba, save_path: Path):
    """Plot score distribution by actual class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        y_proba[y_true == 0], bins=50, alpha=0.6, label="Non-default", density=True, color="#3b82f6"
    )
    ax.hist(
        y_proba[y_true == 1], bins=50, alpha=0.6, label="Default", density=True, color="#ef4444"
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Actual Outcome (Out-of-Fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved score distribution: {save_path}")


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def produce_oof_predictions(model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5):
    """Generate leakage-free out-of-fold predictions for one model.

    A fresh ``TargetEncoder -> classifier`` pipeline is fit per fold on the
    training split only (imported lazily from train_models to share the factory).
    Each row's predicted probability therefore comes from a model that never
    saw that row — a faithful generalization estimate, not a resubstitution one.
    """
    from train_models import make_pipeline

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(X), np.nan, dtype=float)
    use_early_stop = model_name in ("xgboost", "lightgbm")

    from train_models import _eval_set_fit

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline = make_pipeline(model_name, X)
        if use_early_stop:
            _eval_set_fit(pipeline, X_train, y_train, X_val, y_val)
        else:
            pipeline.fit(X_train, y_train)
        oof[val_idx] = pipeline.predict_proba(X_val)[:, 1]

    return oof


def evaluate_oof(y_true, y_proba):
    """Compute metrics from OOF predictions and the optimal-threshold confusion matrix."""
    threshold = find_optimal_threshold(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "ks": float(ks_score(y_true, y_proba)),
        "gini": float(2 * roc_auc_score(y_true, y_proba) - 1),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }

    print("\nOut-of-Fold Metrics (leakage-free):")
    print(f"  AUC:    {metrics['auc']:.4f}")
    print(f"  KS:     {metrics['ks']:.4f}")
    print(f"  Gini:   {metrics['gini']:.4f}")
    print(f"  F1:     {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=FEATURES_TRAIN_CSV)
    args = parser.parse_args()

    print(f"Loading train: {args.train}")
    train_df = pd.read_csv(args.train)

    y = train_df[TARGET_COL]
    X = train_df.drop(columns=[TARGET_COL, "SK_ID_CURR"])

    with open(MODEL_RESULTS_JSON) as f:
        results = json.load(f)
    best_name = results.get("best_model", "lightgbm")
    print(f"Best model from CV: {best_name}")

    # No labeled test set exists (Home Credit application_test.csv has no
    # TARGET). Report leakage-free OUT-OF-FOLD metrics on the training set
    # rather than fabricating a "test AUC" by resubstitution.
    print(
        "No labeled test set available — producing leakage-free OUT-OF-FOLD "
        "predictions for evaluation ..."
    )
    oof_proba = produce_oof_predictions(best_name, X, y)
    y_arr = y.to_numpy()

    metrics = evaluate_oof(y_arr, oof_proba)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(y_arr, oof_proba, IMAGES_DIR / "roc_curve.png")
    plot_pr_curve(y_arr, oof_proba, IMAGES_DIR / "pr_curve.png")
    plot_calibration(y_arr, oof_proba, IMAGES_DIR / "calibration.png")
    plot_score_distribution(y_arr, oof_proba, IMAGES_DIR / "score_distribution.png")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results["oof_metrics"] = metrics
    # Persist OOF predictions for reproducibility / downstream use.
    np.save(REPORTS_DIR / "oof_predictions.npy", oof_proba)
    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUpdated results: {MODEL_RESULTS_JSON}")


if __name__ == "__main__":
    main()
