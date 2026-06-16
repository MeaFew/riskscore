"""Streamlit dashboard for riskscore.

Interactive visualizations:
- Model performance comparison
- ROC / PR curves
- Score distribution
- Feature importance
- SHAP explainability
- Single-case risk calculator
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURES_TEST_CSV,
    FEATURES_TRAIN_CSV,
    IMAGES_DIR,
    MODEL_PATH,
    MODEL_RESULTS_JSON,
    TARGET_COL,
)

st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data
def load_data():
    if not FEATURES_TRAIN_CSV.exists() or not FEATURES_TEST_CSV.exists():
        st.error("Data files not found. Run `make features` first.")
        st.stop()
    train = pd.read_csv(FEATURES_TRAIN_CSV)
    test = pd.read_csv(FEATURES_TEST_CSV)
    if not MODEL_RESULTS_JSON.exists():
        st.error("Model results not found. Run `make train evaluate` first.")
        st.stop()
    with open(MODEL_RESULTS_JSON) as f:
        results = json.load(f)
    return train, test, results


@st.cache_resource
def load_model():
    json_path = MODEL_PATH.with_suffix(".json")
    joblib_path = MODEL_PATH.with_suffix(".joblib")
    if not json_path.exists() and not joblib_path.exists():
        st.error("Model file not found. Run `make train` first.")
        st.stop()
    # Read best model name to choose loader
    with open(MODEL_RESULTS_JSON) as f:
        results = json.load(f)
    best_name = results.get("best_model", "xgboost")
    if best_name == "xgboost" and json_path.exists():
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(json_path))
        return model
    return joblib.load(str(joblib_path))


def main():
    st.title("Credit Risk Scoring Dashboard")
    st.markdown("---")

    train_df, test_df, results = load_data()

    # ── Sidebar ────────────────────────────────────────────────────
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Model Comparison",
            "Score Distribution",
            "Feature Importance",
            "Risk Calculator",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Train: {len(train_df):,} | Test: {len(test_df):,} | Default rate: {train_df[TARGET_COL].mean() * 100:.1f}%"
    )

    # ── Overview ───────────────────────────────────────────────────
    if page == "Overview":
        st.header("Project Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Samples", f"{len(train_df):,}")
        with col2:
            st.metric("Test Samples", f"{len(test_df):,}")
        with col3:
            metadata_cols = sum(1 for c in ["TARGET", "SK_ID_CURR"] if c in train_df.columns)
            st.metric("Features", train_df.shape[1] - metadata_cols)
        with col4:
            st.metric("Default Rate", f"{train_df[TARGET_COL].mean() * 100:.2f}%")

        st.markdown("---")
        st.subheader("Model Performance Summary")

        cv = results.get("cv_results", {})
        data = []
        for name, metrics in cv.items():
            data.append(
                {
                    "Model": name.replace("_", " ").title(),
                    "AUC": f"{metrics['auc_mean']:.4f}",
                    "KS": f"{metrics['ks_mean']:.4f}",
                    "Gini": f"{metrics['gini_mean']:.4f}",
                }
            )
        st.dataframe(pd.DataFrame(data), use_container_width=True)

        st.markdown("---")
        st.subheader("Key Insights")
        st.markdown("""
        - **Class imbalance**: ~8% default rate requires careful handling (SMOTE, class weights, or threshold tuning).
        - **XGBoost / LightGBM** typically outperform logistic regression by 3-5% AUC on this dataset.
        - **KS statistic** > 0.3 is generally considered acceptable for credit scoring; > 0.4 is good.
        - **SHAP analysis** reveals that income-to-credit ratio and external credit scores are the strongest predictors.
        """)

    # ── Model Comparison ───────────────────────────────────────────
    elif page == "Model Comparison":
        st.header("Model Comparison")

        cv = results.get("cv_results", {})
        metrics_choice = st.selectbox("Metric", ["AUC", "KS", "Gini"])
        metric_key = metrics_choice.lower() + "_mean"

        models = list(cv.keys())
        values = [cv[m][metric_key] for m in models]
        errors = [cv[m][metric_key.replace("_mean", "_std")] for m in models]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[m.replace("_", " ").title() for m in models],
                y=values,
                error_y=dict(type="data", array=errors, visible=True),
                marker_color="#3b82f6",
            )
        )
        fig.update_layout(
            title=f"Cross-Validation {metrics_choice}",
            yaxis_title=metrics_choice,
            xaxis_title="Model",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Test metrics
        test_metrics = results.get("test_metrics", {})
        if test_metrics:
            st.subheader("Test Set Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC", f"{test_metrics.get('auc', 0):.4f}")
            c2.metric("KS", f"{test_metrics.get('ks', 0):.4f}")
            c3.metric("Gini", f"{test_metrics.get('gini', 0):.4f}")
            c4.metric("F1", f"{test_metrics.get('f1', 0):.4f}")

    # ── Score Distribution ─────────────────────────────────────────
    elif page == "Score Distribution":
        st.header("Score Distribution")

        model = load_model()
        drop_cols = [c for c in ["SK_ID_CURR", TARGET_COL] if c in test_df.columns]
        X_eval = test_df.drop(columns=drop_cols)
        y_eval = test_df[TARGET_COL]
        y_proba = model.predict_proba(X_eval)[:, 1]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=y_proba[y_eval == 0],
                name="Non-default",
                opacity=0.7,
                nbinsx=50,
                marker_color="#3b82f6",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=y_proba[y_eval == 1],
                name="Default",
                opacity=0.7,
                nbinsx=50,
                marker_color="#ef4444",
            )
        )
        fig.update_layout(
            barmode="overlay",
            title="Predicted Probability Distribution",
            xaxis_title="Default Probability",
            yaxis_title="Count",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix
        threshold = results.get("test_metrics", {}).get("threshold", 0.5)
        y_pred = (y_proba >= threshold).astype(int)
        cm = pd.crosstab(y_eval, y_pred, rownames=["Actual"], colnames=["Predicted"])
        st.subheader("Confusion Matrix")
        st.dataframe(cm, use_container_width=True)

    # ── Feature Importance ─────────────────────────────────────────
    elif page == "Feature Importance":
        st.header("Feature Importance")

        model = load_model()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            features = test_df.drop(
                columns=["SK_ID_CURR", TARGET_COL], errors="ignore"
            ).columns.tolist()
            imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Features (XGBoost)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model does not expose feature_importances_.")

        # SHAP image
        shap_path = IMAGES_DIR / "shap_feature_importance.png"
        if shap_path.exists():
            st.subheader("SHAP Feature Importance")
            st.image(str(shap_path), use_container_width=True)

    # ── Risk Calculator ────────────────────────────────────────────
    elif page == "Risk Calculator":
        st.header("Single-Case Risk Calculator")
        st.markdown("Adjust feature values to see predicted default probability.")

        model = load_model()
        features = test_df.drop(columns=["SK_ID_CURR"], errors="ignore").columns.tolist()

        # Use a random test case as default
        if test_df.empty:
            st.warning("No test data available for risk calculation.")
            st.stop()
        sample_idx = st.number_input("Sample Index", 0, len(test_df) - 1, 0)
        sample = test_df.iloc[sample_idx].drop(labels=[TARGET_COL, "SK_ID_CURR"], errors="ignore")

        user_input = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            with cols[i % 3]:
                default_val = float(sample[feat]) if feat in sample else 0.0
                user_input[feat] = st.number_input(feat, value=default_val, key=feat)

        input_df = pd.DataFrame([user_input])
        proba = model.predict_proba(input_df)[0, 1]

        st.markdown("---")
        st.subheader("Risk Score")

        color = "#ef4444" if proba > 0.5 else "#f59e0b" if proba > 0.2 else "#10b981"
        st.markdown(
            f"<h1 style='color: {color}; text-align: center;'>{proba * 100:.2f}%</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center;'>Predicted default probability</p>",
            unsafe_allow_html=True,
        )

        if proba > 0.5:
            st.error("High risk — recommend rejection or additional collateral.")
        elif proba > 0.2:
            st.warning("Medium risk — recommend manual review.")
        else:
            st.success("Low risk — approve with standard terms.")


if __name__ == "__main__":
    main()
