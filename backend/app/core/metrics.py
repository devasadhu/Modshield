"""
Layer 5 core metrics.

Computes the three metrics the spec says must be kept separate, never
conflated:
  1. Label efficiency — accuracy reached per N labels
  2. Severity-exposure reduction — total severity-weighted items reaching
     humans, system vs naive baseline, at matched accuracy
  3. Calibration quality — ECE before/after calibration

Operates on Orchestrator.history (list[RoundResult]) plus a held-out
evaluation set the orchestrator itself never trains on, since RoundResult
doesn't carry accuracy (it tracks ECE, which is calibration quality, not
classification accuracy — conflating them was explicitly flagged as a
mistake to avoid back in stopping_criterion.py).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from app.models.base import ModelAdapter
from app.core.orchestrator import RoundResult
from app.core.severity_router import RoutingDecision


@dataclass
class LabelEfficiencyPoint:
    cumulative_labels: int
    f1_score: float
    round_num: int


def compute_label_efficiency_curve(
    adapter_snapshots: list[ModelAdapter],
    cumulative_labels: list[int],
    eval_texts: list[str],
    eval_labels: np.ndarray,
) -> list[LabelEfficiencyPoint]:
    """
    adapter_snapshots: one fitted adapter per round (caller must snapshot —
    orchestrator doesn't keep historical model states, only the latest).
    cumulative_labels: total labels used as of that round, same length/order
    as adapter_snapshots.
    eval_texts/eval_labels: a FIXED held-out set, never touched by training,
    used consistently across all rounds and across different strategies so
    comparisons are apples-to-apples.

    Returns one LabelEfficiencyPoint per round, macro-F1 across all 6 labels.
    """
    points = []
    for round_num, (adapter, n_labels) in enumerate(zip(adapter_snapshots, cumulative_labels), start=1):
        probs = adapter.predict_proba(eval_texts)
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(eval_labels, preds, average="macro", zero_division=0)
        points.append(LabelEfficiencyPoint(cumulative_labels=n_labels, f1_score=f1, round_num=round_num))
    return points


@dataclass
class SeverityExposurePoint:
    cumulative_labels: int
    cumulative_human_severity_exposure: float
    round_num: int


def compute_severity_exposure_curve(
    history: list[RoundResult],
    per_round_human_severity: list[float],
) -> list[SeverityExposurePoint]:
    """
    per_round_human_severity: sum of severity_scores for samples that were
    ROUTE_TO_HUMAN or FLAG_AMBIGUOUS that round (i.e. actually reached a
    human) — caller computes this during the round since RoundResult
    doesn't carry per-sample severity, only counts.

    Returns cumulative human severity exposure over cumulative labels used,
    the chart that proves the actual thesis: routed system vs naive
    baseline (caller runs this same function against a naive baseline run
    — e.g. random strategy + always-route-to-human — and plots both curves
    together).
    """
    points = []
    cumulative_labels = 0
    cumulative_exposure = 0.0
    for result, round_severity in zip(history, per_round_human_severity):
        cumulative_labels += result.n_queried
        cumulative_exposure += round_severity
        points.append(SeverityExposurePoint(
            cumulative_labels=cumulative_labels,
            cumulative_human_severity_exposure=cumulative_exposure,
            round_num=result.round_num,
        ))
    return points


def naive_baseline_human_severity(severity_scores: np.ndarray) -> float:
    """
    For the naive baseline comparison: if EVERY queried sample (regardless
    of routing decision) had instead gone straight to a human with no
    routing logic at all, total severity exposure would just be the sum of
    all severity scores for that round. Use this per-round to build the
    "naive" comparison curve via compute_severity_exposure_curve.
    """
    return float(severity_scores.sum())


def calibration_quality_summary(ece_history: list[float]) -> dict:
    """
    Per spec: "ECE before/after temperature scaling, and its effect on
    routing correctness." This function only summarizes the ECE trend
    over rounds (before/after-calibration comparison is computed once
    upstream in calibration.py's expected_calibration_error calls — this
    is just the round-by-round trend view for reporting).
    """
    if not ece_history:
        return {"mean_ece": None, "final_ece": None, "trend": "no data"}

    mean_ece = float(np.mean(ece_history))
    final_ece = ece_history[-1]

    if len(ece_history) >= 2:
        trend = "improving" if ece_history[-1] < ece_history[0] else "worsening"
    else:
        trend = "insufficient data"

    return {
        "mean_ece": mean_ece,
        "final_ece": final_ece,
        "trend": trend,
        "history": list(ece_history),
    }


def exposure_reduction_percent(system_exposure: float, naive_exposure: float) -> float:
    """
    Headline number for the severity-exposure chart: what % of severity-
    weighted human exposure did the system avoid vs. the naive baseline.
    Returns 0.0 if naive_exposure is 0 (no severe content existed at all —
    nothing to reduce, not a system failure).
    """
    if naive_exposure <= 0:
        return 0.0
    return float((1.0 - (system_exposure / naive_exposure)) * 100)