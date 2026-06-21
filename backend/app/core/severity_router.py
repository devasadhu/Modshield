"""
Severity-Aware Router — Layer 3, the core contribution of TriageLoop.

Routing logic, per the spec:

- High calibrated confidence, low epistemic uncertainty, low severity
    -> auto-label
- High calibrated confidence, low epistemic uncertainty, high severity
    -> auto-label, route to periodic audit sample
- High epistemic uncertainty
    -> route to human (model would genuinely benefit from this label)
- High aleatoric uncertainty, low epistemic
    -> flag as inherently ambiguous, route to policy review

This module decides ONLY the routing label. It does NOT do the periodic
random audit sampling (that's audit_sampler.py) or exposure-budget pacing
(that's exposure_budget.py) — kept separate so each piece is independently
testable and the router stays a pure decision function.

Thresholds are configurable, not hardcoded, because the adaptive threshold
logic (adaptive_threshold.py) needs to mutate them between rounds based on
ECE.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RoutingDecision(str, Enum):
    AUTO_LABEL = "auto_label"
    AUTO_LABEL_AUDIT = "auto_label_audit"  # auto-labeled, but flagged for periodic audit
    ROUTE_TO_HUMAN = "route_to_human"
    FLAG_AMBIGUOUS = "flag_ambiguous"


@dataclass
class RouterThresholds:
    confidence_threshold: float = 0.85  # min calibrated confidence to consider auto-labeling
    epistemic_threshold: float = 0.15   # above this -> genuinely uncertain, route to human
    aleatoric_threshold: float = 0.4    # above this (with low epistemic) -> ambiguous
    severity_threshold: float = 0.5     # above this -> needs audit even if auto-labeled


def _max_label_confidence(calibrated_probs: np.ndarray) -> np.ndarray:
    """
    Per-sample confidence = max distance from 0.5 across labels, rescaled
    to [0, 1] where 1.0 = maximally confident (prob near 0 or 1 on at least
    one label) and 0.0 = prob exactly 0.5 on every label.
    """
    return (np.abs(calibrated_probs - 0.5) * 2).max(axis=1)


def route_batch(
    calibrated_probs: np.ndarray,
    epistemic_scores: np.ndarray,
    aleatoric_scores: np.ndarray,
    severity_scores: np.ndarray,
    thresholds: RouterThresholds | None = None,
) -> list[RoutingDecision]:
    """
    All inputs are (n,) arrays except calibrated_probs which is (n, N_LABELS)
    — aligned to the same n samples.

    Returns: list of RoutingDecision, one per sample, in input order.

    Decision precedence (checked in this order per sample):
      1. High epistemic uncertainty -> ROUTE_TO_HUMAN
         (checked first: if the model is genuinely ignorant about this
         sample, that overrides everything else, including severity)
      2. High aleatoric, low epistemic -> FLAG_AMBIGUOUS
      3. High confidence, low epistemic, high severity -> AUTO_LABEL_AUDIT
      4. High confidence, low epistemic, low severity -> AUTO_LABEL
      5. Fallback (low confidence but also low epistemic/aleatoric signal —
         an edge case the heuristics didn't clearly cover) -> ROUTE_TO_HUMAN,
         conservative default rather than silently auto-labeling something
         the confidence check didn't clear.
    """
    thresholds = thresholds or RouterThresholds()
    confidence = _max_label_confidence(calibrated_probs)
    n = calibrated_probs.shape[0]

    decisions = []
    for i in range(n):
        epi = epistemic_scores[i]
        ale = aleatoric_scores[i]
        sev = severity_scores[i]
        conf = confidence[i]

        if epi > thresholds.epistemic_threshold:
            decisions.append(RoutingDecision.ROUTE_TO_HUMAN)
            continue

        if ale > thresholds.aleatoric_threshold:
            decisions.append(RoutingDecision.FLAG_AMBIGUOUS)
            continue

        if conf >= thresholds.confidence_threshold:
            if sev >= thresholds.severity_threshold:
                decisions.append(RoutingDecision.AUTO_LABEL_AUDIT)
            else:
                decisions.append(RoutingDecision.AUTO_LABEL)
            continue

        decisions.append(RoutingDecision.ROUTE_TO_HUMAN)

    return decisions


def routing_summary(decisions: list[RoutingDecision]) -> dict:
    """Quick counts per decision type, for round-by-round logging/dashboards."""
    summary = {d.value: 0 for d in RoutingDecision}
    for d in decisions:
        summary[d.value] += 1
    return summary