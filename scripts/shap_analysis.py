"""SHAP interpretability analysis for riskscore.

Generates:
- SHAP summary plot (beeswarm)
- SHAP dependence plots for top features
- SHAP force plot for a single prediction (saved as HTML)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURES_TEST_CSV,  # only used as argparse default for --test
    FEATURES_TRAIN_CSV,
    IMAGES_DIR,
    MODEL_PATH,
    MODEL_RESULTS_JSON,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_SUMMARY_PNG,
    TARGET_COL,
)


def load_model():
    """Load the best trained model, auto-detecting file format."""
    # Read best model name from results
    with open(MODEL_RESULTS_JSON) as f:
        results = json.load(f)
    best_name = results.get("best_model", "xgboost")

    json_path = MODEL_PATH.with_suffix(".json")
    joblib_path = MODEL_PATH.with_suffix(".joblib")
    if best_name == "xgboost" and json_path.exists():
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(json_path))
        return model
    elif joblib_path.exists():
        return joblib.load(str(joblib_path))
    else:
        # Last resort: any model file in MODELS_DIR
        candidates = list(MODEL_PATH.parent.glob("*_risk_model.*"))
        if candidates:
            p = candidates[0]
            # Only attempt XGBoost native loading for .json if best_model is xgboost
            if best_name == "xgboost" and p.suffix == ".json":
                import xgboost as xgb

                model = xgb.XGBClassifier()
                model.load_model(str(p))
                return model
            elif p.suffix == ".joblib":
                return joblib.load(str(p))
            else:
                # Non-xgboost .json files can't be loaded here; prefer .joblib
                raise FileNotFoundError(f"No compatible model file found for stem {MODEL_PATH}")
        raise FileNotFoundError(f"No model file found for stem {MODEL_PATH}")


def plot_shap_summary(shap_values, X_sample, feature_names, save_path: Path):
    """Generate SHAP beeswarm summary plot."""
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    fig = plt.gcf()  # shap.summary_plot creates its own figure internally
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP summary: {save_path}")


def plot_top_features(shap_values, X_sample, feature_names, top_n: int = 10, save_dir: Path = None):
    """Plot SHAP bar chart of top features."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, mean_abs_shap[top_idx], align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Feature Importance (SHAP)")
    fig.tight_layout()

    if save_dir:
        save_path = save_dir / "shap_feature_importance.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance: {save_path}")
    plt.close(fig)

    return [feature_names[i] for i in top_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=FEATURES_TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=FEATURES_TEST_CSV)
    parser.add_argument(
        "--sample", type=int, default=1000, help="Number of samples for SHAP computation"
    )
    args = parser.parse_args()

    print("Loading data ...")
    train_df = pd.read_csv(args.train)
    X_train = train_df.drop(columns=[TARGET_COL, "SK_ID_CURR"])

    print("Loading model ...")
    model = load_model()
    # Model is already trained — use loaded parameters directly for SHAP

    # Sample for SHAP computation (too slow on full data)
    X_sample = X_train.sample(n=min(args.sample, len(X_train)), random_state=RANDOM_STATE)

    print(f"Computing SHAP values on {len(X_sample)} training samples ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, TreeExplainer may return a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use class 1 (default)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = X_sample.columns.tolist()

    # Summary plot
    print("Generating SHAP summary plot ...")
    plot_shap_summary(shap_values, X_sample, feature_names, SHAP_SUMMARY_PNG)

    # Top features bar chart
    print("Generating feature importance plot ...")
    top_features = plot_top_features(
        shap_values, X_sample, feature_names, top_n=10, save_dir=IMAGES_DIR
    )

    # Dependence plots for top 3 features
    print("Generating dependence plots ...")
    for i, feat in enumerate(top_features[:3]):
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat,
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            ax=ax,
        )
        fig.tight_layout()
        save_path = IMAGES_DIR / f"shap_dependence_{i + 1}_{feat[:30]}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved dependence plot: {save_path}")

    # Force plot for a single high-risk case
    print("Generating force plot for single case ...")
    shap_force = shap.force_plot(
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
        shap_values[0],
        X_sample.iloc[0],
        feature_names=feature_names,
        show=False,
    )
    force_path = REPORTS_DIR / "shap_force_plot.html"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shap.save_html(str(force_path), shap_force)
    print(f"Saved force plot: {force_path}")

    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    main()
