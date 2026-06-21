"""
Severity estimator — independent of the toxicity classifier (Layer 1's
adapter). Outputs a single scalar severity_estimate in [0, 1] per sample,
used by the router (auto/audit/human/ambiguous decision) and by
cost-weighted scoring (Layer 2) and exposure-budget tracking (this layer).

Two components, combined:

1. Heuristic severity weights — not all 6 Jigsaw labels are equally severe.
   threat/identity_hate/severe_toxic are weighted higher than
   obscene/insult/toxic. This is a documented, debatable judgment call —
   not empirically derived — and should be called out as such in the
   paper/writeup (per the spec's "How do you know your severity estimator
   is accurate?" pushback: it's explicitly a proxy, not solved).

2. Calibration-aware adjustment — severity computed from raw probabilities
   is only as trustworthy as those probabilities. This estimator expects
   CALIBRATED probabilities as input (i.e. already passed through
   TemperatureScaler/PlattScaler from Layer 1's calibration module) so a
   confidently-wrong model doesn't silently produce a falsely low or high
   severity estimate.

A separate "classifier" mode is included for cases where you want to learn
severity weights from data (e.g. human-annotated severity scores) rather
than hand-set heuristic weights — swap in once any such labels exist.
"""

import numpy as np

from app.models.base import LABELS

# Heuristic per-label severity weights, in LABELS order:
# [toxic, severe_toxic, obscene, threat, insult, identity_hate]
# Rationale: threat and identity_hate carry the highest psychological cost
# to a reviewer; severe_toxic is explicitly an escalation of toxic in the
# Jigsaw taxonomy. obscene/insult/toxic are comparatively milder.
DEFAULT_SEVERITY_WEIGHTS = {
    "toxic": 0.3,
    "severe_toxic": 0.8,
    "obscene": 0.4,
    "threat": 1.0,
    "insult": 0.4,
    "identity_hate": 0.9,
}


class HeuristicSeverityEstimator:
    """
    severity_estimate[i] = max over labels of (calibrated_prob[i, label] *
    severity_weight[label])

    Uses max rather than weighted sum deliberately: a sample that's only
    confidently flagged as "threat" should count as high-severity even if
    every other label is near zero — severity shouldn't get diluted by
    label sparsity the way a sum would.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_SEVERITY_WEIGHTS
        # build weight vector aligned to LABELS order once, not per call
        self._weight_vec = np.array([self.weights[label] for label in LABELS])

    def estimate(self, calibrated_probs: np.ndarray) -> np.ndarray:
        """
        calibrated_probs: (n, N_LABELS) — MUST be post-calibration
        probabilities, not raw model output.

        Returns: (n,) severity estimate in [0, 1]
        """
        weighted = calibrated_probs * self._weight_vec[None, :]
        return weighted.max(axis=1)


class LearnedSeverityEstimator:
    """
    Placeholder for a data-driven severity estimator, to swap in once
    human-annotated severity labels exist (out of scope for v1 per the
    project spec — no real moderator deployment). For now this just wraps
    HeuristicSeverityEstimator so the interface is stable and callers don't
    need to change when a real learned version is built later.
    """

    def __init__(self, fallback: HeuristicSeverityEstimator | None = None):
        self.fallback = fallback or HeuristicSeverityEstimator()
        self._fitted = False

    def fit(self, calibrated_probs: np.ndarray, severity_labels: np.ndarray) -> None:
        raise NotImplementedError(
            "No human-annotated severity labels exist yet (out of scope for v1). "
            "Use HeuristicSeverityEstimator until a labeled severity dataset exists."
        )

    def estimate(self, calibrated_probs: np.ndarray) -> np.ndarray:
        return self.fallback.estimate(calibrated_probs)