"""Feature engineering for credit-risk-scoring.

Implements:
- WOE (Weight of Evidence) binning for numeric features
- IV (Information Value) filtering for feature selection
- Cross-features (ratios, differences, EXT_SOURCE interactions)
- Aggregate features from external tables (simplified)

References:
- Siddiqi, N. (2012). Credit Risk Scorecards.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLEANED_TEST_CSV,
    CLEANED_TRAIN_CSV,
    FEATURES_TEST_CSV,
    FEATURES_TRAIN_CSV,
    IV_THRESHOLD,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    WOE_MAX_BINS,
)

warnings.filterwarnings("ignore")


def compute_woe_iv(df: pd.DataFrame, feature: str, target: str, n_bins: int = WOE_MAX_BINS) -> tuple[pd.DataFrame, float]:
    """Compute WOE and IV for a single numeric feature using quantile binning.

    Returns:
        woe_df: DataFrame with bins, WOE, IV per bin
        iv: total Information Value
    """
    df = df[[feature, target]].copy()
    df = df.dropna()

    # Quantile-based binning
    try:
        df["bin"] = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    except ValueError:
        # Too many duplicate values — fallback to fewer bins
        unique_vals = df[feature].nunique()
        n_bins = min(n_bins, max(2, unique_vals // 2))
        df["bin"] = pd.qcut(df[feature], q=n_bins, duplicates="drop")

    # Compute WOE per bin
    grouped = df.groupby("bin", observed=False)[target].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    # Smoothing to avoid division by zero
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()
    grouped["good_dist"] = (grouped["good"] + 0.5) / (total_good + 0.5)
    grouped["bad_dist"] = (grouped["bad"] + 0.5) / (total_bad + 0.5)

    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
    grouped["iv"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    iv = grouped["iv"].sum()
    return grouped[["woe", "iv"]], iv



def compute_iv_all(df: pd.DataFrame, numeric_cols: list[str], target: str = TARGET_COL) -> pd.DataFrame:
    """Compute IV for all numeric features."""
    results = []
    for col in numeric_cols:
        if col == target:
            continue
        try:
            _, iv = compute_woe_iv(df, col, target)
            results.append({"feature": col, "iv": iv})
        except (ValueError, KeyError, IndexError):
            warnings.warn(f"IV computation skipped for {col}")
            results.append({"feature": col, "iv": 0.0})
    return pd.DataFrame(results).sort_values("iv", ascending=False)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature engineering pipeline."""
    print("Building features ...")

    train = train_df.copy()
    test = test_df.copy()

    y = train[TARGET_COL].copy()
    train = train.drop(columns=[TARGET_COL])

    # ── 1. Basic ratio features ────────────────────────────────────
    print("  Creating ratio features ...")
    if "AMT_CREDIT" in train.columns and "AMT_INCOME_TOTAL" in train.columns:
        train["CREDIT_TO_INCOME_RATIO"] = train["AMT_CREDIT"] / (train["AMT_INCOME_TOTAL"] + 1)
        test["CREDIT_TO_INCOME_RATIO"] = test["AMT_CREDIT"] / (test["AMT_INCOME_TOTAL"] + 1)

    if "AMT_ANNUITY" in train.columns and "AMT_INCOME_TOTAL" in train.columns:
        train["ANNUITY_TO_INCOME_RATIO"] = train["AMT_ANNUITY"] / (train["AMT_INCOME_TOTAL"] + 1)
        test["ANNUITY_TO_INCOME_RATIO"] = test["AMT_ANNUITY"] / (test["AMT_INCOME_TOTAL"] + 1)

    if "AMT_CREDIT" in train.columns and "AMT_ANNUITY" in train.columns:
        train["CREDIT_TO_ANNUITY_RATIO"] = train["AMT_CREDIT"] / (train["AMT_ANNUITY"] + 1)
        test["CREDIT_TO_ANNUITY_RATIO"] = test["AMT_CREDIT"] / (test["AMT_ANNUITY"] + 1)

    if "DAYS_BIRTH" in train.columns:
        train["AGE_YEARS"] = -train["DAYS_BIRTH"] / 365.25
        test["AGE_YEARS"] = -test["DAYS_BIRTH"] / 365.25

    if "DAYS_EMPLOYED" in train.columns and "DAYS_BIRTH" in train.columns:
        train["EMPLOYED_TO_AGE_RATIO"] = train["DAYS_EMPLOYED"] / (train["DAYS_BIRTH"] + 1)
        test["EMPLOYED_TO_AGE_RATIO"] = test["DAYS_EMPLOYED"] / (test["DAYS_BIRTH"] + 1)

    # ── 2. EXT_SOURCE interactions ─────────────────────────────────
    ext_cols = [c for c in train.columns if c.startswith("EXT_SOURCE_")]
    if len(ext_cols) >= 2:
        print(f"  Creating EXT_SOURCE interactions ({len(ext_cols)} sources) ...")
        for i, c1 in enumerate(ext_cols):
            for c2 in ext_cols[i + 1:]:
                train[f"{c1}_x_{c2}"] = train[c1] * train[c2]
                test[f"{c1}_x_{c2}"] = test[c1] * test[c2]
            train[f"{c1}_squared"] = train[c1] ** 2
            test[f"{c1}_squared"] = test[c1] ** 2

    # ── 3. Categorical encoding ────────────────────────────────────
    cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
    print(f"  Encoding {len(cat_cols)} categorical columns ...")

    # Combine for consistent encoding
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    n_train = len(train)

    for col in cat_cols:
        # Target encoding with smoothing
        global_mean = y.mean()
        smoothing = 10.0
        agg = pd.DataFrame({col: train[col], TARGET_COL: y}).groupby(col)[TARGET_COL].agg(["mean", "count"])
        agg["smooth"] = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        mapping = agg["smooth"].to_dict()
        combined[f"{col}_TE"] = combined[col].map(mapping).fillna(global_mean)

    # One-hot for low-cardinality
    low_card = [c for c in cat_cols if combined[c].nunique() <= 5]
    remaining_cat = [c for c in cat_cols if c not in low_card]
    if low_card:
        combined = pd.get_dummies(combined, columns=low_card, drop_first=True)

    # Drop remaining original categoricals
    combined = combined.drop(columns=remaining_cat)

    # ── 4. IV-based feature selection ──────────────────────────────
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("SK_ID_CURR",)]

    print(f"  Computing IV for {len(numeric_cols)} numeric features ...")
    combined_train = combined.iloc[:n_train].copy()
    combined_train[TARGET_COL] = y.values

    iv_df = compute_iv_all(combined_train, numeric_cols, TARGET_COL)
    selected_features = iv_df[iv_df["iv"] >= IV_THRESHOLD]["feature"].tolist()
    print(f"  Selected {len(selected_features)} / {len(numeric_cols)} features (IV >= {IV_THRESHOLD})")
    print(f"  Top 5 by IV: {iv_df.head(5).to_dict('records')}")

    # Keep SK_ID_CURR + selected + target
    keep_cols = ["SK_ID_CURR"] + selected_features
    if TARGET_COL in combined.columns:
        keep_cols.append(TARGET_COL)

    train_out = combined.iloc[:n_train][keep_cols].copy()
    test_out = combined.iloc[n_train:][keep_cols].copy()
    train_out[TARGET_COL] = y.values

    return train_out, test_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=CLEANED_TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=CLEANED_TEST_CSV)
    args = parser.parse_args()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_features, test_features = build_features(train_df, test_df)

    train_features.to_csv(FEATURES_TRAIN_CSV, index=False)
    test_features.to_csv(FEATURES_TEST_CSV, index=False)

    print(f"\nSaved: {FEATURES_TRAIN_CSV} ({train_features.shape})")
    print(f"Saved: {FEATURES_TEST_CSV} ({test_features.shape})")


if __name__ == "__main__":
    main()
