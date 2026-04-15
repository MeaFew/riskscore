"""Data preprocessing pipeline for credit-risk-scoring.

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


def cap_outliers(df: pd.DataFrame, numeric_cols: list[str], lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
    """Cap extreme outliers at given quantiles."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            low, high = df[col].quantile([lower_q, upper_q])
            df[col] = df[col].clip(low, high)
    return df


def handle_missing_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Impute missing numeric values with median."""
    df = df.copy()
    imputer = SimpleImputer(strategy="median")
    missing_cols = [c for c in numeric_cols if c in df.columns and df[c].isna().any()]
    if missing_cols:
        df[missing_cols] = imputer.fit_transform(df[missing_cols])
    return df


def handle_missing_categorical(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Fill missing categorical values with 'MISSING'."""
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING")
    return df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline."""
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

    # Combine for consistent preprocessing
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_train = len(train_df)

    # Outlier capping
    print("Capping outliers ...")
    combined = cap_outliers(combined, numeric_cols)

    # Missing value handling
    print("Imputing missing values ...")
    combined = handle_missing_numeric(combined, numeric_cols)
    combined = handle_missing_categorical(combined, cat_cols)

    # Split back
    train_clean = combined.iloc[:n_train].copy()
    test_clean = combined.iloc[n_train:].copy()

    # Reattach target
    train_clean[TARGET_COL] = y.values

    return train_clean, test_clean


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
    print(f"Default rate: {train_clean[TARGET_COL].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
