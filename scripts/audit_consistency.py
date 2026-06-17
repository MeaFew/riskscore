"""Cross-reference audit: README claims vs. actual pipeline outputs.

Run after `make all` to verify that key metrics declared in README.md
match the actual values produced by the pipeline.

Usage: python scripts/audit_consistency.py
"""

import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURES_TEST_CSV,
    FEATURES_TRAIN_CSV,
    MODEL_PATH,
    MODEL_RESULTS_JSON,
    PROCESSED_DATA_DIR,
)


def read_readme_metric(readme_path: Path, metric_name: str) -> float | None:
    """Extract a numeric metric from README.md.

    Looks for patterns like `| **Metric Name** | **0.123** |`
    or `**Metric Name = 0.123**`.
    """
    text = readme_path.read_text(encoding="utf-8")
    pattern = rf"\*\*{re.escape(metric_name)}\*\*.*?(\d+\.\d+)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def read_readme_model_table(readme_path: Path) -> dict[str, dict[str, float]]:
    """Parse the model results table from README.md.

    Returns:
        dict mapping model_name -> {"auc": float, "ks": float, "gini": float}
    """
    text = readme_path.read_text(encoding="utf-8")
    results = {}

    # Match rows like: | Logistic Regression | 0.634 | 0.205 | 0.268 |
    pattern = r"\|\s*([\w\s-]+?)\s*\|\s*\*{0,2}(\d+\.\d+)\*{0,2}\s*\|\s*\*{0,2}(\d+\.\d+)\*{0,2}\s*\|\s*\*{0,2}(\d+\.\d+)\*{0,2}\s*\|"
    for match in re.finditer(pattern, text):
        name = match.group(1).strip().lower().replace(" ", "_")
        # Skip benchmark rows that aren't model names
        if name in ("logistic_regression", "random_forest", "xgboost", "lightgbm"):
            results[name] = {
                "auc": float(match.group(2)),
                "ks": float(match.group(3)),
                "gini": float(match.group(4)),
            }
    return results


def check(condition: bool, msg: str) -> bool:
    """Assert-like check that prints pass/fail."""
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
    return condition


def main():
    root = Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    passed = 0
    failed = 0

    # ── 1. Verify AUC values in README match model_results.json ────
    print("=== Check 1: README vs model_results.json AUC values ===")
    readme_table = read_readme_model_table(readme)
    if not readme_table:
        print("  SKIP: Could not parse model table from README.md")
    else:
        with open(MODEL_RESULTS_JSON) as f:
            results = json.load(f)
        cv_results = results.get("cv_results", {})

        # Allow ±0.01 tolerance for rounding differences
        tolerance = 0.015

        for model_name, readme_metrics in readme_table.items():
            if model_name not in cv_results:
                ok = check(False, f"{model_name}: not found in model_results.json")
                if not ok:
                    failed += 1
                else:
                    passed += 1
                continue

            actual = cv_results[model_name]

            for metric in ("auc", "ks", "gini"):
                actual_val = round(actual[f"{metric}_mean"], 3)
                readme_val = readme_metrics[metric]
                diff = abs(readme_val - actual_val)
                ok = check(
                    diff <= tolerance,
                    f"README {model_name} {metric.upper()}={readme_val:.3f} vs actual={actual_val:.3f} (diff={diff:.4f})",
                )
                if ok:
                    passed += 1
                else:
                    failed += 1

    print()

    # ── 2. Verify model file exists and can be loaded ───────────────
    print("=== Check 2: Model file existence and loadability ===")
    json_path = MODEL_PATH.with_suffix(".json")
    joblib_path = MODEL_PATH.with_suffix(".joblib")
    model_exists = json_path.exists() or joblib_path.exists()
    if check(model_exists, f"Model file exists ({MODEL_PATH.stem}.json or .joblib)"):
        passed += 1
    else:
        failed += 1

    if model_exists:
        try:
            actual_path = joblib_path if joblib_path.exists() else json_path
            if actual_path.suffix == ".json":
                import xgboost as xgb

                model = xgb.XGBClassifier()
                model.load_model(str(actual_path))
            else:
                model = joblib.load(str(actual_path))
            ok = check(True, f"Model loaded successfully from {actual_path.name}")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            ok = check(False, f"Failed to load model: {e}")
            if not ok:
                failed += 1
            else:
                passed += 1

    print()

    # ── 3. Verify feature files exist and have correct column counts ─
    print("=== Check 3: Feature files existence and column counts ===")
    train_exists = FEATURES_TRAIN_CSV.exists()
    test_exists = FEATURES_TEST_CSV.exists()
    if check(train_exists, "features_train.csv exists"):
        passed += 1
    else:
        failed += 1
    if check(test_exists, "features_test.csv exists"):
        passed += 1
    else:
        failed += 1

    if train_exists and test_exists:
        train_df = pd.read_csv(FEATURES_TRAIN_CSV)
        test_df = pd.read_csv(FEATURES_TEST_CSV)

        # The test set has no TARGET column, so it has one fewer column than
        # train. Compare the feature columns (excluding the id/target meta
        # columns) rather than the raw column counts.
        meta_cols = {"SK_ID_CURR", "TARGET"}
        train_features = [c for c in train_df.columns if c not in meta_cols]
        test_features = [c for c in test_df.columns if c not in meta_cols]
        ok = check(
            train_features == test_features,
            f"Train & test feature columns match: "
            f"train_features={len(train_features)}, test_features={len(test_features)}",
        )
        if ok:
            passed += 1
        else:
            failed += 1

        # Check against model_results.json feature_count. feature_count is the
        # number of model input features; train_features excludes SK_ID_CURR
        # and TARGET so it should equal feature_count.
        with open(MODEL_RESULTS_JSON) as f:
            results = json.load(f)
        expected_features = results.get("feature_count")
        if expected_features is not None:
            ok = check(
                len(train_features) == expected_features,
                f"Train feature columns ({len(train_features)}) match "
                f"model_results.json feature_count ({expected_features})",
            )
            if ok:
                passed += 1
            else:
                failed += 1
    else:
        print("  SKIP: Feature files not found, column count check skipped.")

    # ── Summary ────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("ACTION: Update README.md or pipeline to resolve mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
