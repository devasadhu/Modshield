"""
Query-by-Committee (QBC) strategy.

Trains K committee members on bootstrap resamples of the labeled pool,
then selects unlabeled samples where the committee disagrees most.
Disagreement = mean pairwise variance across labels (same metric as
qbc_epistemic_estimate in uncertainty_decomposition.py — these bootstrap
heads double as a second, independent epistemic estimate at no extra cost,
per the spec).

Model-agnostic: takes a model_factory callable so it works with either
LogRegAdapter or DistilBertAdapter without modification. For DistilBERT,
expect this to be slow (K full fine-tunes) — fine for eval sweeps on
LogReg first, revisit committee size before running on DistilBERT.
"""

import numpy as np

from app.core.query_strategies import QueryStrategy
from app.core.uncertainty_decomposition import qbc_epistemic_estimate


class QueryByCommitteeStrategy(QueryStrategy):
    name = "qbc"

    def __init__(self, model_factory, n_committee: int = 5, seed: int | None = None):
        """
        model_factory: callable() -> ModelAdapter, returns a fresh untrained
        adapter instance (e.g. lambda: LogRegAdapter())
        """
        self.model_factory = model_factory
        self.n_committee = n_committee
        self.rng = np.random.default_rng(seed)
        self.last_disagreement: np.ndarray | None = None  # cached for reuse/inspection

    def _train_committee(self, train_texts: list[str], train_labels: np.ndarray):
        n = len(train_texts)
        members = []
        for _ in range(self.n_committee):
            boot_idx = self.rng.choice(n, size=n, replace=True)
            boot_texts = [train_texts[i] for i in boot_idx]
            boot_labels = train_labels[boot_idx]
            model = self.model_factory()
            model.fit(boot_texts, boot_labels)
            members.append(model)
        return members

    def select(self, probs: np.ndarray, k: int, **kwargs) -> np.ndarray:
        """
        QBC needs training data + unlabeled texts, which don't fit the
        plain (probs, k) signature used by entropy/margin/random. Use
        select_with_committee() instead; this override exists only to
        satisfy the abstract interface and fails loudly if misused.
        """
        raise NotImplementedError(
            "QueryByCommitteeStrategy needs train_texts/train_labels/unlabeled_texts — "
            "call select_with_committee() instead of select()."
        )

    def select_with_committee(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        unlabeled_texts: list[str],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns: (selected_indices, disagreement_scores)
        selected_indices are local indices into unlabeled_texts.
        disagreement_scores is the full per-sample score array, also handed
        to Layer 3's router as a second epistemic estimate.
        """
        members = self._train_committee(train_texts, train_labels)
        committee_probs = [m.predict_proba(unlabeled_texts) for m in members]

        disagreement = qbc_epistemic_estimate(committee_probs)
        self.last_disagreement = disagreement

        k = min(k, len(disagreement))
        selected = np.argsort(disagreement)[::-1][:k]  # highest disagreement first
        return selected, disagreement