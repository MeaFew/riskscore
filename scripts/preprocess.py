"""Data preprocessing pipeline for riskscore.

Handles missing values, outlier capping, categorical encoding,
and saves cleaned train/test sets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLEANED_TEST_CSV,
    CLEANED_TRAIN_CSV,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test data."""
    print(f"Loading train from {train_path} ...")
    train_df = pd.read_csv(train_path)
    print(f"  Train shape: {train_df.shape}")

    print(f"Loading test from {test_path} ...")
    test_df = pd.read_csv(test_path)
    print(f"  Test shape: {test_df.shape}")

    return train_df, test_df


def cap_outliers(
    df: pd.DataFrame, numeric_cols: list[str], lower_q: float = 0.001, upper_q: float = 0.999
) -> pd.DataFrame:
    """Cap extreme outliers at given quantiles.

    Quantiles are estimated from the frame passed in. Callers that want a
    leakage-free fit/transform should pass ``fit_quantiles=True`` first on the
    train frame to capture the boundaries, then apply them to test via
    :func:`cap_outliers_with_bounds`.
    """
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            low, high = df[col].quantile([lower_q, upper_q])
            df[col] = df[col].clip(low, high)
    return df


def compute_outlier_bounds(
    df: pd.DataFrame, numeric_cols: list[str], lower_q: float = 0.001, upper_q: float = 0.999
) -> dict[str, tuple[float, float]]:
    """Estimate outlier-capping boundaries from a (train) frame only.

    Returns a mapping ``{col: (low, high)}`` that can be applied to any frame
    via :func:`cap_outliers_with_bounds`, so test rows never influence the
    boundaries used on train rows (leakage fix).
    """
    bounds: dict[str, tuple[float, float]] = {}
    for col in numeric_cols:
        if col in df.columns:
            low, high = df[col].quantile([lower_q, upper_q])
            bounds[col] = (float(low), float(high))
    return bounds


def cap_outliers_with_bounds(
    df: pd.DataFrame, bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Apply precomputed outlier boundaries to a frame."""
    df = df.copy()
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(low, high)
    return df


def handle_missing_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Impute missing numeric values with median (fit_transform on the given frame).

    For leakage-free imputation where the median must come from train only, use
    :func:`fit_missing_numeric` + :func:`transform_missing_numeric`.
    """
    df = df.copy()
    imputer = SimpleImputer(strategy="median")
    missing_cols = [c for c in numeric_cols if c in df.columns and df[c].isna().any()]
    if missing_cols:
        df[missing_cols] = imputer.fit_transform(df[missing_cols])
    return df


def fit_missing_numeric(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[SimpleImputer, list[str]]:
    """Fit a median imputer on a (train) frame only.

    Returns ``(imputer, fitted_cols)`` so :func:`transform_missing_numeric`
    can supply the exact column order/name set the imputer was fit on
    (sklearn validates feature names and rejects mismatches).
    """
    imputer = SimpleImputer(strategy="median")
    cols = [c for c in numeric_cols if c in df.columns]
    if cols:
        imputer.fit(df[cols])
    return imputer, cols


def transform_missing_numeric(
    df: pd.DataFrame, numeric_cols: list[str], imputer: SimpleImputer
) -> pd.DataFrame:
    """Apply a fitted imputer to a frame (used for the test set).

    Transforms only the columns present in ``df`` that the imputer knows about,
    supplying them in the imputer's fitted name order. Rows/columns with no
    missing values pass through unchanged.
    """
    df = df.copy()
    # Use the imputer's fitted feature names when available, intersected with
    # what's present in this frame, so sklearn's name validation is satisfied.
    fitted = getattr(imputer, "feature_names_in_", None)
    if fitted is not None:
        cols = [c for c in fitted if c in df.columns]
    else:
        cols = [c for c in numeric_cols if c in df.columns]
    if cols:
        has_missing = [c for c in cols if df[c].isna().any()]
        if has_missing:
            # Transform the full fitted subset present, write back only needed.
            transformed = imputer.transform(df[cols])
            df[cols] = transformed
    return df


def handle_missing_categorical(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Fill missing categorical values with 'MISSING'."""
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING")
    return df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline.

    Leakage-free: all fit-able statistics (outlier boundaries, median imputation)
    are estimated from the **train** frame only, then applied to test. The two
    frames are no longer concatenated before fitting, so test rows cannot
    influence train-row preprocessing.
    """
    # Separate target
    y = train_df[TARGET_COL].copy()
    train_df = train_df.drop(columns=[TARGET_COL])

    # Align columns
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    # Identify column types
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in train_df.columns if c not in numeric_cols and c != "SK_ID_CURR"]

    # Remove ID from processing lists
    if "SK_ID_CURR" in numeric_cols:
        numeric_cols.remove("SK_ID_CURR")

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")

    # Outlier capping — boundaries from train only, applied to both
    print("Capping outliers (fit on train only) ...")
    bounds = compute_outlier_bounds(train_df, numeric_cols)
    train_df = cap_outliers_with_bounds(train_df, bounds)
    test_df = cap_outliers_with_bounds(test_df, bounds)

    # Missing value handling — imputer fit on train only
    print("Imputing missing values (fit on train only) ...")
    imputer, _fitted_cols = fit_missing_numeric(train_df, numeric_cols)
    train_df = transform_missing_numeric(train_df, numeric_cols, imputer)
    test_df = transform_missing_numeric(test_df, numeric_cols, imputer)

    # Categorical fill is target-free, so safe on each frame independently
    train_df = handle_missing_categorical(train_df, cat_cols)
    test_df = handle_missing_categorical(test_df, cat_cols)

    # Reattach target
    train_df[TARGET_COL] = y.values

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=TEST_CSV)
    args = parser.parse_args()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args.train, args.test)
    train_clean, test_clean = preprocess(train_df, test_df)

    train_clean.to_csv(CLEANED_TRAIN_CSV, index=False)
    test_clean.to_csv(CLEANED_TEST_CSV, index=False)

    print(f"\nSaved: {CLEANED_TRAIN_CSV} ({train_clean.shape})")
    print(f"Saved: {CLEANED_TEST_CSV} ({test_clean.shape})")
    print(f"Default rate: {train_clean[TARGET_COL].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
