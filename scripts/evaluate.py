"""Model evaluation and calibration for riskscore.

Generates:
- ROC curve plot
- Precision-Recall curve plot
- Calibration plot (reliability diagram)
- Score distribution plot
- Confusion matrix at optimal threshold
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_utils import ks_score

from config import (
    FEATURES_TEST_CSV,
    FEATURES_TRAIN_CSV,
    IMAGES_DIR,
    MODEL_PATH,
    MODEL_RESULTS_JSON,
    MODELS_DIR,
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
    ax.set_title("ROC Curve")
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
    ax.set_title("Precision-Recall Curve")
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
    ax.set_title("Calibration Plot (Reliability Diagram)")
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
    ax.set_title("Score Distribution by Actual Outcome")
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


def evaluate_model(model, X_test, y_test):
    """Full evaluation pipeline."""
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import roc_auc_score as sk_auc

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "auc": float(sk_auc(y_test, y_proba)),
        "ks": float(ks_score(y_test, y_proba)),
        "gini": float(2 * sk_auc(y_test, y_proba) - 1),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }

    print("\nTest Set Metrics:")
    print(f"  AUC:    {metrics['auc']:.4f}")
    print(f"  KS:     {metrics['ks']:.4f}")
    print(f"  Gini:   {metrics['gini']:.4f}")
    print(f"  F1:     {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.4f}")

    # Plots
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(y_test, y_proba, IMAGES_DIR / "roc_curve.png")
    plot_pr_curve(y_test, y_proba, IMAGES_DIR / "pr_curve.png")
    plot_calibration(y_test, y_proba, IMAGES_DIR / "calibration.png")
    plot_score_distribution(y_test, y_proba, IMAGES_DIR / "score_distribution.png")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=FEATURES_TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=FEATURES_TEST_CSV)
    args = parser.parse_args()

    print(f"Loading train: {args.train}")
    train_df = pd.read_csv(args.train)
    print(f"Loading test: {args.test}")
    test_df = pd.read_csv(args.test)

    y_train = train_df[TARGET_COL]
    X_train = train_df.drop(columns=[TARGET_COL, "SK_ID_CURR"])

    if TARGET_COL in test_df.columns:
        y_test = test_df[TARGET_COL]
        X_test = test_df.drop(columns=[TARGET_COL, "SK_ID_CURR"])
    else:
        # No target in test — split train for evaluation
        print("Test set has no TARGET - splitting train 80/20 for evaluation ...")
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )

    # Load best model from CV results (read JSON once)
    with open(MODEL_RESULTS_JSON) as f:
        results = json.load(f)
    best_name = results.get("best_model", "xgboost")

    # Resolve the persisted model file via the shared helper (kept consistent
    # with shap_analysis.py and the dashboard).
    from metrics_utils import find_best_model_path

    model_path = find_best_model_path(MODELS_DIR, MODEL_PATH.stem, best_name)

    if best_name == "xgboost" and model_path.suffix == ".json":
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
    else:
        model = joblib.load(str(model_path))
    print(f"Loaded model from {model_path}")

    metrics = evaluate_model(model, X_test, y_test)

    # Save results (reuse already-loaded results dict)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results["test_metrics"] = metrics
    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUpdated results: {MODEL_RESULTS_JSON}")


if __name__ == "__main__":
    main()
