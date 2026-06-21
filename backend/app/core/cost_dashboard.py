"""
Cost dashboard — Layer 5.

Per spec: "$-per-label and severity-weighted harm cost, total saved vs
baseline." Thin aggregation layer over cost_weighting.py (Layer 2) and
metrics.py (this layer) — no new cost model here, just summarizes what
those already compute into the reporting numbers a stakeholder would
actually want to see.

"$" here is whatever unit base_cost/severity_weight were configured in
(literal currency if you have real per-label review costs, or an arbitrary
unit if not — this module doesn't assume one or the other).
"""

from dataclasses import dataclass

import numpy as np

from app.core.cost_weighting import expected_label_cost


@dataclass
class CostDashboardSummary:
    total_labels: int
    total_cost: float
    avg_cost_per_label: float
    total_severity_weighted_harm_cost: float  # cost attributable to severity alone, base_cost stripped out
    naive_baseline_total_cost: float
    total_saved: float
    percent_saved: float


def compute_cost_dashboard(
    severity_scores: np.ndarray,
    routed_to_human_mask: np.ndarray,
    base_cost: float = 1.0,
    severity_weight: float = 1.0,
) -> CostDashboardSummary:
    """
    severity_scores: (n,) severity estimate per queried sample, this run
    routed_to_human_mask: (n,) bool — True if that sample actually reached
    a human (ROUTE_TO_HUMAN or FLAG_AMBIGUOUS); auto-labeled samples cost
    only the (much smaller, here assumed ~0) automation cost, not the full
    human review cost.

    "Naive baseline" = every queried sample sent to a human regardless of
    routing (no Layer 3 router at all) — this is the comparison point the
    spec's "total saved vs baseline" refers to.
    """
    n = len(severity_scores)
    per_sample_cost = expected_label_cost(severity_scores, base_cost, severity_weight)

    # actual cost: only items that reached a human incur the full cost;
    # auto-labeled items are treated as ~free (automation cost not modeled
    # separately here — extend with a small flat auto-label cost if needed)
    actual_cost_per_sample = np.where(routed_to_human_mask, per_sample_cost, 0.0)
    total_cost = float(actual_cost_per_sample.sum())

    severity_only_cost = severity_weight * np.clip(severity_scores, 0, 1)
    total_severity_harm_cost = float(np.where(routed_to_human_mask, severity_only_cost, 0.0).sum())

    naive_baseline_total_cost = float(per_sample_cost.sum())  # everything routed to human

    total_saved = naive_baseline_total_cost - total_cost
    percent_saved = (total_saved / naive_baseline_total_cost * 100) if naive_baseline_total_cost > 0 else 0.0

    return CostDashboardSummary(
        total_labels=n,
        total_cost=total_cost,
        avg_cost_per_label=(total_cost / n) if n > 0 else 0.0,
        total_severity_weighted_harm_cost=total_severity_harm_cost,
        naive_baseline_total_cost=naive_baseline_total_cost,
        total_saved=total_saved,
        percent_saved=float(percent_saved),
    )