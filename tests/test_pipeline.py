"""Tests for credit-risk-scoring preprocessing and feature engineering."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_COL
from scripts.generate_mock_data import generate_application_train
from scripts.preprocess import cap_outliers, handle_missing_categorical, handle_missing_numeric, preprocess
from scripts.feature_engineering import build_features, compute_woe_iv


@pytest.fixture
def mock_train_df():
    return generate_application_train(n_rows=1000, seed=42)


@pytest.fixture
def mock_test_df():
    df = generate_application_train(n_rows=200, seed=43)
    return df.drop(columns=[TARGET_COL])


class TestPreprocess:
    def test_load_mock_data(self, mock_train_df):
        assert len(mock_train_df) == 1000
        assert TARGET_COL in mock_train_df.columns
        assert mock_train_df[TARGET_COL].isin([0, 1]).all()

    def test_cap_outliers(self, mock_train_df):
        numeric = mock_train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in ("SK_ID_CURR", TARGET_COL)]
        df = cap_outliers(mock_train_df, numeric, lower_q=0.01, upper_q=0.99)
        for col in numeric:
            assert df[col].min() >= mock_train_df[col].quantile(0.01)
            assert df[col].max() <= mock_train_df[col].quantile(0.99)

    def test_handle_missing_numeric(self, mock_train_df):
        numeric = mock_train_df.select_dtypes(include=[np.number]).columns.tolist()
        df = mock_train_df.copy()
        df.loc[0:10, numeric[0]] = np.nan
        df = handle_missing_numeric(df, numeric)
        assert df[numeric[0]].isna().sum() == 0

    def test_handle_missing_categorical(self, mock_train_df):
        cat = mock_train_df.select_dtypes(include=["object"]).columns.tolist()
        df = mock_train_df.copy()
        if cat:
            df.loc[0:5, cat[0]] = np.nan
            df = handle_missing_categorical(df, cat)
            assert df[cat[0]].isna().sum() == 0
            assert (df[cat[0]] == "MISSING").sum() == 6

    def test_preprocess_pipeline(self, mock_train_df, mock_test_df):
        train_clean, test_clean = preprocess(mock_train_df, mock_test_df)
        assert TARGET_COL in train_clean.columns
        assert TARGET_COL not in test_clean.columns
        assert train_clean.isna().sum().sum() == 0
        assert test_clean.isna().sum().sum() == 0


class TestFeatureEngineering:
    def test_build_features(self, mock_train_df, mock_test_df):
        train_clean, test_clean = preprocess(mock_train_df, mock_test_df)
        train_features, test_features = build_features(train_clean, test_clean)
        assert TARGET_COL in train_features.columns
        assert "SK_ID_CURR" in train_features.columns
        assert len(train_features) == len(train_clean)

    def test_woe_iv_computation(self, mock_train_df):
        numeric = mock_train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in ("SK_ID_CURR", TARGET_COL)]
        if numeric:
            woe_df, iv = compute_woe_iv(mock_train_df, numeric[0], TARGET_COL)
            assert iv >= 0
            assert "woe" in woe_df.columns
            assert "iv" in woe_df.columns


class TestEndToEnd:
    def test_full_pipeline(self, mock_train_df, mock_test_df):
        train_clean, test_clean = preprocess(mock_train_df, mock_test_df)
        train_features, test_features = build_features(train_clean, test_clean)
        assert train_features.shape[0] == mock_train_df.shape[0]
        assert test_features.shape[0] == mock_test_df.shape[0]


class TestFeatureOutputValidity:
    """P2-12: Tests that cover meaningful pipeline validation."""

    def test_build_features_has_expected_columns(self, mock_train_df, mock_test_df):
        """Feature engineering output must include SK_ID_CURR, TARGET, and ratio features."""
        train_clean, test_clean = preprocess(mock_train_df, mock_test_df)
        train_features, test_features = build_features(train_clean, test_clean)

        assert "SK_ID_CURR" in train_features.columns
        assert "SK_ID_CURR" in test_features.columns
        assert TARGET_COL in train_features.columns
        assert TARGET_COL not in test_features.columns

        # Ratio features should be created when source columns exist
        if "AMT_CREDIT" in train_clean.columns and "AMT_INCOME_TOTAL" in train_clean.columns:
            assert "CREDIT_TO_INCOME_RATIO" in train_features.columns
        if "DAYS_BIRTH" in train_clean.columns:
            assert "AGE_YEARS" in train_features.columns

        # EXT_SOURCE interaction features
        source_cols = [c for c in train_features.columns if c.startswith("EXT_SOURCE_")]
        assert len(source_cols) > 0, "Should have at least some EXT_SOURCE columns"

    def test_default_rate_in_valid_range(self, mock_train_df):
        """Training data default rate must be between 0% and 20%."""
        default_rate = mock_train_df[TARGET_COL].mean()
        assert 0.0 <= default_rate <= 0.20, (
            f"Default rate {default_rate:.4f} outside valid range [0, 0.20]"
        )

    def test_preprocessed_data_no_nan_in_key_columns(self, mock_train_df, mock_test_df):
        """Preprocessed training data must have no NaN in numeric key columns."""
        train_clean, test_clean = preprocess(mock_train_df, mock_test_df)

        key_cols = [
            c for c in ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
                         "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1",
                         "EXT_SOURCE_2", "EXT_SOURCE_3"]
            if c in train_clean.columns
        ]

        for col in key_cols:
            assert train_clean[col].isna().sum() == 0, f"NaN found in {col} after preprocessing"
            assert test_clean[col].isna().sum() == 0, f"NaN found in {col} after preprocessing"
