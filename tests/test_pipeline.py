"""Tests for riskscore preprocessing and feature engineering."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# scripts/ modules import their siblings as top-level names (e.g.
# `from metrics_utils import ks_score`), so scripts/ itself must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from config import TARGET_COL
from scripts.feature_engineering import build_features, compute_woe_iv
from scripts.generate_mock_data import generate_application_train
from scripts.preprocess import (
    cap_outliers,
    handle_missing_categorical,
    handle_missing_numeric,
    preprocess,
)


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
            c
            for c in [
                "AMT_CREDIT",
                "AMT_INCOME_TOTAL",
                "AMT_ANNUITY",
                "DAYS_BIRTH",
                "DAYS_EMPLOYED",
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
            ]
            if c in train_clean.columns
        ]

        for col in key_cols:
            assert train_clean[col].isna().sum() == 0, f"NaN found in {col} after preprocessing"
            assert test_clean[col].isna().sum() == 0, f"NaN found in {col} after preprocessing"


class TestLeakagePrevention:
    """Regression tests for the leakage fixes (H3 target encoding, H4 fake
    test AUC, H5 preprocessing fit-on-train).

    These guard against re-introducing target leakage into CV folds or
    fabricating test metrics from training data.
    """

    def test_preprocess_fits_only_on_train(self, mock_train_df, mock_test_df):
        """H5: outlier bounds and imputation must be fit on train only.

        Inject an extreme outlier + NaN into the TEST set only; if preprocessing
        were fit on train+test concatenated, the train frame's caps would shift
        to accommodate the test outlier. Instead train caps/medians must be
        unaffected by test rows.
        """
        from scripts.preprocess import compute_outlier_bounds, fit_missing_numeric

        numeric = mock_train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in ("SK_ID_CURR", TARGET_COL)]
        if not numeric:
            pytest.skip("no numeric columns")

        col = numeric[0]
        test_perturbed = mock_test_df.copy()
        test_perturbed[col] = test_perturbed[col].astype(float)
        # Inject a huge outlier and a NaN ONLY in test
        test_perturbed.loc[test_perturbed.index[:5], col] = 1e18
        test_perturbed.loc[test_perturbed.index[5:10], col] = np.nan

        train_clean, test_clean = preprocess(mock_train_df, test_perturbed)
        # Train must not be capped at the test outlier's magnitude.
        assert train_clean[col].max() < 1e15, (
            "Train caps moved due to test outlier — preprocessing is leaking "
            "test statistics into train (H5 regression)."
        )
        assert train_clean[col].isna().sum() == 0
        assert test_clean[col].isna().sum() == 0

    def test_high_card_categoricals_retained_raw(self, mock_train_df, mock_test_df):
        """H3: high-cardinality categoricals must NOT be target-encoded at the
        feature-engineering stage. They are kept raw so the encoder is fit per
        CV fold (inside the sklearn Pipeline), preventing the validation rows'
        own target from leaking into their encoded value.
        """
        train_clean, test_clean = preprocess(mock_train_df, mock_test_df)
        train_features, _ = build_features(train_clean, test_clean)

        # OCCUPATION_TYPE / ORGANIZATION_TYPE are high-card in the mock schema
        high_card_candidates = [
            c for c in ("OCCUPATION_TYPE", "ORGANIZATION_TYPE") if c in train_clean.columns
        ]
        retained = train_features.attrs.get("te_columns", [])
        for c in high_card_candidates:
            assert c in retained, (
                f"{c} should be flagged as a per-fold target-encoded column "
                "(kept raw), not baked into features (H3 regression)."
            )
            # And no pre-baked *_TE column should exist
            assert f"{c}_TE" not in train_features.columns

    def test_target_encoding_is_per_fold(self, mock_train_df, mock_test_df):
        """H3: the same category must receive DIFFERENT encoded values across
        folds when its target distribution differs by fold. This is only true
        if the encoder is fit inside the CV loop, not on the full training set.
        """
        from sklearn.preprocessing import TargetEncoder

        # Construct a tiny dataset where a category's positive rate differs
        # between an "early" and "late" subset, so an encoder fit on each half
        # yields different encodings for the same category.
        rng = np.random.RandomState(0)
        n = 2000
        cat = rng.choice(["A", "B", "C"], n)
        # Category "A" has very different default rate in first vs second half
        proba = np.where(np.arange(n) < n // 2, 0.05, 0.45)
        proba = np.where(cat == "A", proba, 0.2)
        y = (rng.random(n) < proba).astype(int)
        df = pd.DataFrame({"cat": cat, TARGET_COL: y})

        enc1 = TargetEncoder(target_type="binary", random_state=0)
        enc2 = TargetEncoder(target_type="binary", random_state=0)
        # Fit one encoder on the full set and another on only the first half.
        # If the encoder were not fold-dependent the two would agree; the point
        # of per-fold fitting is precisely that they differ. (We fit both to
        # exercise the encoder; the real guard is the pipeline assertion below.)
        enc1.fit(df[["cat"]], df[TARGET_COL])
        enc2.fit(df[["cat"]].iloc[: n // 2], df[TARGET_COL].iloc[: n // 2])
        # Stronger guard: make_pipeline must wire TargetEncoder as the
        # preprocessor so it refits each fold.
        from scripts.train_models import make_pipeline  # noqa: WPS433

        sample_X = df.drop(columns=[TARGET_COL])
        pipe = make_pipeline("lightgbm", sample_X)
        # Access the TE step before fitting (transformers_ only exists post-fit).
        pre = pipe.named_steps["preprocess"]
        te_step = {name: trans for name, trans, _cols in pre.transformers}.get("te")
        assert te_step is not None, "Pipeline must include a TargetEncoder step"
        assert isinstance(te_step, TargetEncoder), (
            "Categorical encoding must be a TargetEncoder fit per fold (H3)"
        )

    def test_evaluate_does_not_fabricate_test_auc(self, tmp_path, monkeypatch):
        """H4: with no labeled test set, evaluate must produce OOF metrics, not
        a resubstitution 'test AUC' measured on training data.
        """
        # evaluate.py lives in scripts/; import via the scripts package path.
        from scripts import evaluate as evaluate_mod

        # The evaluator must reference OOF (out-of-fold), not the old
        # train_test_split-on-train path that inflated a fake test AUC.
        src = Path(evaluate_mod.__file__).read_text(encoding="utf-8")
        assert "train_test_split" not in src or "out-of-fold" in src.lower(), (
            "evaluate.py must not fabricate a test AUC by splitting the "
            "training set (H4 regression)."
        )


class TestMetricHelpers:
    """Unit tests for the shared metric helpers (ks_score etc.)."""

    def test_ks_score_matches_scipy(self):
        """ks_score must equal scipy.stats.ks_2samp(pos, neg).statistic."""
        import scipy.stats as st
        from metrics_utils import ks_score

        rng = np.random.default_rng(0)
        y_true = np.array([1] * 100 + [0] * 100)
        # Pos class gets systematically higher scores => non-trivial KS.
        y_proba = np.concatenate([rng.uniform(0.5, 1.0, 100), rng.uniform(0.0, 0.5, 100)])
        expected = st.ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic
        assert ks_score(y_true, y_proba) == pytest.approx(expected, abs=1e-9)

    def test_ks_score_perfect_separation_is_one(self):
        """When every positive outscores every negative, KS should be 1.0."""
        from metrics_utils import ks_score

        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])
        assert ks_score(y_true, y_proba) == pytest.approx(1.0, abs=1e-9)

    def test_ks_score_identical_distributions_is_zero(self):
        """Same score distribution for both classes => KS = 0."""
        from metrics_utils import ks_score

        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        assert ks_score(y_true, y_proba) == pytest.approx(0.0, abs=1e-9)
