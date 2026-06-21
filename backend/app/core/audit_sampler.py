"""
Periodic audit mechanism.

Per the spec: "randomly re-route X% of auto-labeled items to a human
anyway, track agreement rate as a silent-degradation check."

Two responsibilities:
1. select_audit_sample — pick which auto-labeled items get re-routed
2. record_agreement / agreement_rate — track how often the human agrees
   with what the model auto-labeled, over a rolling history. A falling
   agreement rate is the early-warning signal that auto-labeling has
   silently started failing (model drift, distribution shift, etc.) even
   though confidence/calibration metrics alone wouldn't necessarily catch it.

Distinct from RoutingDecision.AUTO_LABEL_AUDIT (severity_router.py), which
flags high-severity auto-labels for audit at routing time. This module
additionally audits a random X% of ALL auto-labeled items (including
AUTO_LABEL, not just AUTO_LABEL_AUDIT) — severity-triggered audits and
random-sample audits are complementary, not the same mechanism.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AuditRecord:
    sample_id: int
    round_num: int
    model_label: np.ndarray   # what auto-labeling assigned
    human_label: np.ndarray   # what the human assigned on audit
    agreed: bool


class AuditSampler:
    def __init__(self, audit_fraction: float = 0.05, history_window: int = 200, seed: int | None = None):
        """
        audit_fraction: fraction of auto-labeled items to randomly re-route
        per round (the "X%" in the spec).
        history_window: how many recent audit records to keep for the
        rolling agreement_rate calculation.
        """
        self.audit_fraction = audit_fraction
        self.history_window = history_window
        self.rng = np.random.default_rng(seed)
        self.records: list[AuditRecord] = []

    def select_audit_sample(self, auto_labeled_ids: np.ndarray) -> np.ndarray:
        """
        auto_labeled_ids: sample IDs that were auto-labeled this round
        (any RoutingDecision starting with AUTO_LABEL — caller filters).

        Returns: subset of auto_labeled_ids selected for human audit.
        """
        n = len(auto_labeled_ids)
        n_audit = max(1, int(round(n * self.audit_fraction))) if n > 0 else 0
        if n_audit == 0:
            return np.array([], dtype=auto_labeled_ids.dtype if n > 0 else int)
        n_audit = min(n_audit, n)
        return self.rng.choice(auto_labeled_ids, size=n_audit, replace=False)

    def record_audit_result(
        self,
        sample_id: int,
        round_num: int,
        model_label: np.ndarray,
        human_label: np.ndarray,
        match_threshold: float = 0.5,
    ) -> AuditRecord:
        """
        Compares model_label vs human_label (binarized at match_threshold
        if labels are probabilities rather than already-binary) and stores
        the result. Exact match across all labels = agreed.
        """
        model_bin = (model_label >= match_threshold).astype(int)
        human_bin = (human_label >= match_threshold).astype(int)
        agreed = bool(np.array_equal(model_bin, human_bin))

        record = AuditRecord(
            sample_id=sample_id,
            round_num=round_num,
            model_label=model_label,
            human_label=human_label,
            agreed=agreed,
        )
        self.records.append(record)
        if len(self.records) > self.history_window:
            self.records = self.records[-self.history_window:]
        return record

    def agreement_rate(self, last_n: int | None = None) -> float | None:
        """
        Rolling agreement rate over the last `last_n` audit records (or all
        retained records if None). Returns None if there's no audit history
        yet (avoid pretending we have a meaningful rate from zero samples).
        """
        records = self.records if last_n is None else self.records[-last_n:]
        if not records:
            return None
        return sum(r.agreed for r in records) / len(records)

    def is_degrading(self, threshold: float = 0.90, last_n: int = 50) -> bool:
        """
        Silent-degradation flag: True if recent agreement rate has dropped
        below `threshold`. Caller (orchestrator) should react by, e.g.,
        tightening the adaptive confidence threshold or pausing auto-labeling.
        """
        rate = self.agreement_rate(last_n=last_n)
        if rate is None:
            return False  # not enough data to claim degradation
        return rate < threshold