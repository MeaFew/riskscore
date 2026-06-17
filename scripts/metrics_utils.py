"""Shared metric utilities for credit-risk scoring.

Kept separate from config.py so that importing the config module (paths and
hyperparameters only) does not pull in scipy as a transitive dependency.
"""

from __future__ import annotations

import scipy.stats as _scipy_stats


def ks_score(y_true, y_proba) -> float:
    """Compute the Kolmogorov-Smirnov statistic between score distributions.

    A higher KS indicates better separation between the positive (default) and
    negative (repaid) score distributions. Standard credit-risk discrimination
    metric alongside AUC/Gini.
    """
    pos = y_proba[y_true == 1]
    neg = y_proba[y_true == 0]
    return _scipy_stats.ks_2samp(pos, neg).statistic


def find_best_model_path(models_dir, model_stem, best_name: str = "xgboost"):
    """Locate the persisted best-model file under ``models_dir``.

    Resolution order (deterministic): for XGBoost prefer the native ``.json``
    checkpoint, for other models prefer ``.joblib``; then fall back to a
    ``{best_name}_risk_model.*`` naming convention, then to any
    ``*_risk_model.*`` file in the directory. Centralized here so evaluate.py,
    shap_analysis.py, and the dashboard all agree on the chosen file.

    Returns the resolved Path, or raises FileNotFoundError.
    """
    ext_order = (".json", ".joblib") if best_name == "xgboost" else (".joblib", ".json")
    for ext in ext_order:
        candidate = models_dir / f"{model_stem}{ext}"
        if candidate.exists():
            return candidate
    for ext in ext_order:
        candidate = models_dir / f"{best_name}_risk_model{ext}"
        if candidate.exists():
            return candidate
    candidates = sorted(models_dir.glob("*_risk_model.*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No model file found in {models_dir}")
