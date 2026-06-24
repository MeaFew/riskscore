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
