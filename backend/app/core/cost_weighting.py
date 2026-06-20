"""
Cost-weighted scoring.

Per the spec: "each strategy's raw score divided by expected_label_cost
(base cost + severity-weighted cost), so query priority reflects
information gain per dollar, not just information gain."

This wraps a raw score array (from entropy/margin/QBC-disagreement/BADGE
gradient-norm/etc.) rather than replacing any strategy — every strategy
above can opt into cost-weighting by passing its raw per-sample scores
through here before taking top-k.

expected_label_cost is intentionally simple and pluggable: base_cost is a
flat per-label cost (time/money), severity_cost scales with how severe the
content is estimated to be (more severe content costs more — reviewer
fatigue/psychological cost, not just time). severity_estimate comes from
Layer 3's severity estimator once it exists; until then, pass zeros to get
plain base-cost weighting.
"""

import numpy as np


def expected_label_cost(
    severity_estimate: np.ndarray,
    base_cost: float = 1.0,
    severity_weight: float = 1.0,
) -> np.ndarray:
    """
    severity_estimate: (n,) values in [0, 1], 0 = benign, 1 = most severe.
    Returns: (n,) cost per sample. Always >= base_cost, never zero, so
    division in cost_weighted_scores never divides by zero.
    """
    severity_estimate = np.clip(severity_estimate, 0.0, 1.0)
    return base_cost + severity_weight * severity_estimate


def cost_weighted_scores(
    raw_scores: np.ndarray,
    severity_estimate: np.ndarray | None = None,
    base_cost: float = 1.0,
    severity_weight: float = 1.0,
) -> np.ndarray:
    """
    raw_scores: (n,) any strategy's per-sample information-gain score
    (higher = more informative; e.g. entropy, QBC disagreement, BADGE norm)

    severity_estimate: (n,) or None. If None, severity contributes nothing
    and this just divides by a flat base_cost (no-op rescaling, useful
    before the severity estimator exists in Layer 3).

    Returns: (n,) cost-adjusted scores. Same ranking semantics as input
    (higher = pick first) — caller's existing argsort/top-k logic still
    applies unchanged.
    """
    n = len(raw_scores)
    if severity_estimate is None:
        severity_estimate = np.zeros(n)

    costs = expected_label_cost(severity_estimate, base_cost, severity_weight)
    return raw_scores / costs


def select_top_k_cost_weighted(
    raw_scores: np.ndarray,
    k: int,
    severity_estimate: np.ndarray | None = None,
    base_cost: float = 1.0,
    severity_weight: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper: cost-weight then return top-k indices, highest first."""
    weighted = cost_weighted_scores(raw_scores, severity_estimate, base_cost, severity_weight)
    k = min(k, len(weighted))
    return np.argsort(weighted)[::-1][:k]