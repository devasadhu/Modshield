"""
Uncertainty decomposition via MC Dropout.

Given N stochastic forward passes (DistilBertAdapter.mc_dropout_predict),
split total predictive uncertainty into:

- Aleatoric: average entropy across passes — uncertainty inherent to the
  data itself (label noise/ambiguity). More labels won't reduce this.
- Epistemic: total entropy of the mean prediction minus aleatoric —
  uncertainty due to model ignorance. This is what the model would
  actually benefit from seeing more labels for.

This is the standard mutual-information decomposition (BALD-style), applied
per-label then aggregated, since this is a multi-label problem.

Used downstream by Layer 3's router to distinguish "route to human, the
model needs this" (high epistemic) from "flag as ambiguous, more labels
won't help" (high aleatoric, low epistemic).
"""

import numpy as np


def _binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def decompose_uncertainty(mc_probs: np.ndarray) -> dict:
    """
    mc_probs: (n_passes, n_samples, N_LABELS) — output of mc_dropout_predict

    Returns dict with per-sample, per-label arrays of shape (n_samples, N_LABELS):
      - mean_probs: average prediction across passes
      - total_uncertainty: entropy of the mean prediction
      - aleatoric: mean of per-pass entropies
      - epistemic: total_uncertainty - aleatoric (mutual information)

    And per-sample aggregated (mean across labels) versions for convenience:
      - epistemic_score, aleatoric_score (n_samples,)
    """
    mean_probs = mc_probs.mean(axis=0)  # (n_samples, N_LABELS)

    total_uncertainty = _binary_entropy(mean_probs)  # (n_samples, N_LABELS)

    per_pass_entropy = _binary_entropy(mc_probs)  # (n_passes, n_samples, N_LABELS)
    aleatoric = per_pass_entropy.mean(axis=0)  # (n_samples, N_LABELS)

    epistemic = total_uncertainty - aleatoric
    # Numerical noise can push this slightly negative; clip to zero.
    epistemic = np.clip(epistemic, 0, None)

    return {
        "mean_probs": mean_probs,
        "total_uncertainty": total_uncertainty,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "epistemic_score": epistemic.mean(axis=1),
        "aleatoric_score": aleatoric.mean(axis=1),
    }


def qbc_epistemic_estimate(committee_probs: list[np.ndarray]) -> np.ndarray:
    """
    Second, independent epistemic estimate from query-by-committee
    disagreement (bootstrap-sampled heads), per the spec: "QBC's bootstrap
    heads double as a second epistemic estimate at no extra cost."

    committee_probs: list of (n_samples, N_LABELS) arrays, one per committee
    member (e.g. from QueryByCommitteeStrategy's bootstrap heads).

    Returns: (n_samples,) disagreement score, mean pairwise variance across
    labels — same shape/semantics as epistemic_score above, so the router
    can combine or cross-check the two estimates.
    """
    stacked = np.stack(committee_probs, axis=0)  # (n_members, n_samples, N_LABELS)
    variance = stacked.var(axis=0)  # (n_samples, N_LABELS)
    return variance.mean(axis=1)  # (n_samples,)