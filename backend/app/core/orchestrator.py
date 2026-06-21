"""
Orchestrator — runs the full active learning cycle, wiring together every
layer built so far:

  QUERY    -> Layer 2 strategy picks which unlabeled samples to act on
  ROUTE    -> Layer 3 router decides auto-label / audit / human / ambiguous
  LABEL    -> auto-label directly, or simulate a human via Layer 4 personas
  RETRAIN  -> Layer 1 adapter refits on the updated labeled pool
  EVALUATE -> Layer 1 calibration + Layer 5 metrics (label-efficiency etc.)
  repeat, until a stopping criterion fires (see stopping_criterion.py)

This is intentionally synchronous and in-process — no Celery/Redis here.
Per the build-order plan: validate the loop end-to-end first, add async
infra only after this works. Designed to be swapped into an async task
later without changing the step logic itself, since each step is already
a self-contained method.

Round-by-round logging is appended to self.history, consumed by Layer 5's
label-efficiency / severity-exposure curves.
"""

from dataclasses import dataclass, field

import numpy as np

from app.models.base import ModelAdapter, N_LABELS
from app.db.pool_manager import PoolManager
from app.core.calibration import TemperatureScaler, expected_calibration_error
from app.core.severity_estimator import HeuristicSeverityEstimator
from app.core.severity_router import route_batch, RouterThresholds, RoutingDecision, routing_summary
from app.core.adaptive_threshold import AdaptiveThresholdController
from app.core.audit_sampler import AuditSampler
from app.core.annotator_personas import SimulatedAnnotator


@dataclass
class RoundResult:
    round_num: int
    n_queried: int
    n_auto_labeled: int
    n_routed_to_human: int
    n_flagged_ambiguous: int
    ece: float
    confidence_threshold: float
    audit_agreement_rate: float | None
    routing_summary: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        pool: PoolManager,
        adapter: ModelAdapter,
        query_strategy,                  # any QueryStrategy with .select(probs, k)
        annotator: SimulatedAnnotator,    # simulates the human in ROUTE_TO_HUMAN/FLAG_AMBIGUOUS
        batch_size: int = 50,
        max_rounds: int = 20,
        severity_estimator: HeuristicSeverityEstimator | None = None,
        adaptive_controller: AdaptiveThresholdController | None = None,
        audit_sampler: AuditSampler | None = None,
        thresholds: RouterThresholds | None = None,
        calibration_holdout_frac: float = 0.2,
    ):
        self.pool = pool
        self.adapter = adapter
        self.query_strategy = query_strategy
        self.annotator = annotator
        self.batch_size = batch_size
        self.max_rounds = max_rounds

        self.severity_estimator = severity_estimator or HeuristicSeverityEstimator()
        self.adaptive_controller = adaptive_controller or AdaptiveThresholdController()
        self.audit_sampler = audit_sampler or AuditSampler()
        self.thresholds = thresholds or RouterThresholds()
        self.calibration_holdout_frac = calibration_holdout_frac

        self.calibrator = TemperatureScaler()
        self.history: list[RoundResult] = []
        self._adapter_trained = False

    def run(self) -> list[RoundResult]:
        # If a seed labeled set already exists (cold-start handled upstream),
        # fit once before round 1 so the first QUERY step isn't cold.
        if len(self.pool.labeled_ids()) > 0 and not self._adapter_trained:
            seed_texts, seed_labels = self.pool.get_labeled_data()
            self.adapter.fit(seed_texts, seed_labels)
            self._adapter_trained = True
            self._refresh_calibration(seed_texts, seed_labels)

        for _ in range(self.max_rounds):
            if len(self.pool.unlabeled_ids()) == 0:
                break
            result = self.step()
            self.history.append(result)
        return self.history

    def step(self) -> RoundResult:
        self.pool.next_round()
        round_num = self.pool.round_num

        # -- QUERY --
        unlabeled_ids = self.pool.unlabeled_ids()
        unlabeled_texts = self.pool.get_texts(unlabeled_ids)

        if self._adapter_trained:
            raw_probs = self.adapter.predict_proba(unlabeled_texts)
        else:
            # cold start: no model yet, uniform probs so entropy/margin
            # strategies degrade gracefully to ~random selection
            raw_probs = np.full((len(unlabeled_ids), N_LABELS), 0.5)

        k = min(self.batch_size, len(unlabeled_ids))
        local_query_idx = self.query_strategy.select(raw_probs, k)
        query_ids = unlabeled_ids[local_query_idx]
        query_texts = self.pool.get_texts(query_ids)
        query_probs = raw_probs[local_query_idx]

        # -- calibrate confidence for routing --
        calibrated_probs = self._calibrate_safe(query_probs)

        # -- uncertainty + severity signals --
        # Without MC dropout wired in yet (DistilBERT-only feature), use a
        # margin-based proxy for epistemic and a flat low aleatoric so the
        # router still has *some* signal with LogRegAdapter. Swap for real
        # decompose_uncertainty() output once running on DistilBertAdapter.
        epistemic_proxy = 1.0 - np.abs(calibrated_probs - 0.5).max(axis=1) * 2  # high when near 0.5
        aleatoric_proxy = np.zeros(len(query_ids))
        severity_scores = self.severity_estimator.estimate(calibrated_probs)

        adaptive_thresholds = self.adaptive_controller.apply(self.thresholds)
        decisions = route_batch(calibrated_probs, epistemic_proxy, aleatoric_proxy, severity_scores, adaptive_thresholds)

        # -- ROUTE + LABEL --
        auto_ids, human_ids = [], []
        for qid, decision, probs_row in zip(query_ids, decisions, calibrated_probs):
            if decision in (RoutingDecision.AUTO_LABEL, RoutingDecision.AUTO_LABEL_AUDIT):
                auto_ids.append(qid)
            else:  # ROUTE_TO_HUMAN or FLAG_AMBIGUOUS both go to the simulated human for v0
                human_ids.append(qid)

        if auto_ids:
            auto_ids = np.array(auto_ids)
            auto_labels = (calibrated_probs[np.isin(query_ids, auto_ids)] >= 0.5).astype(int)
            self.pool.submit_labels(auto_ids, auto_labels, auto=True)

        if human_ids:
            human_ids = np.array(human_ids)
            true_labels = self.pool.get_true_labels(human_ids)  # simulation ground truth
            human_labels = self.annotator.label_batch(true_labels)
            self.pool.submit_labels(human_ids, human_labels, auto=False)

        # -- periodic audit on auto-labeled items --
        audit_ids = self.audit_sampler.select_audit_sample(np.array(auto_ids) if len(auto_ids) else np.array([], dtype=int))
        for aid in audit_ids:
            model_label = self.pool.labels[aid]
            true_label = self.pool.get_true_labels(np.array([aid]))[0]
            human_relabel = self.annotator.label_batch(true_label[None, :])[0]
            self.audit_sampler.record_audit_result(int(aid), round_num, model_label, human_relabel)

        # -- RETRAIN --
        train_texts, train_labels = self.pool.get_labeled_data()
        self.adapter.fit(train_texts, train_labels)
        self._adapter_trained = True

        # -- EVALUATE (calibration check on a fresh holdout slice of labeled data) --
        ece = self._refresh_calibration(train_texts, train_labels)
        new_threshold = self.adaptive_controller.update(ece)

        return RoundResult(
            round_num=round_num,
            n_queried=len(query_ids),
            n_auto_labeled=len(auto_ids) if len(auto_ids) else 0,
            n_routed_to_human=sum(1 for d in decisions if d == RoutingDecision.ROUTE_TO_HUMAN),
            n_flagged_ambiguous=sum(1 for d in decisions if d == RoutingDecision.FLAG_AMBIGUOUS),
            ece=ece,
            confidence_threshold=new_threshold,
            audit_agreement_rate=self.audit_sampler.agreement_rate(),
            routing_summary=routing_summary(decisions),
        )

    def _calibrate_safe(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator.temperatures is None:
            return probs  # not calibrated yet (round 0) — pass through raw
        return self.calibrator.transform(probs)

    def _refresh_calibration(self, train_texts: list[str], train_labels: np.ndarray) -> float:
        n = len(train_texts)
        if n < 10:
            return 1.0  # not enough data yet for a meaningful ECE
        holdout_n = max(5, int(n * self.calibration_holdout_frac))
        holdout_texts = train_texts[-holdout_n:]
        holdout_labels = train_labels[-holdout_n:]

        holdout_probs = self.adapter.predict_proba(holdout_texts)
        self.calibrator.fit(holdout_probs, holdout_labels)
        calibrated = self.calibrator.transform(holdout_probs)
        return expected_calibration_error(calibrated, holdout_labels)