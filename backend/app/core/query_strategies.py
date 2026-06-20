"""
Query strategy engine — decides which unlabeled samples go to a human next.

Common interface: .select(probs, k) -> indices (into the unlabeled pool,
not global sample_ids — caller maps back via PoolManager.unlabeled_ids()).

Starting with three strategies to validate the orchestrator loop end-to-end:
- RandomStrategy: baseline, no information used
- EntropyStrategy: classic uncertainty sampling
- MarginStrategy: smallest gap between top-2 predicted classes

Multi-label note: probs has shape (n_unlabeled, N_LABELS). Per-sample score
is aggregated across labels (mean here) — revisit this if a per-label or
max-label aggregation proves more informative once real data is in.

Diversity (core-set/k-center), QBC, BADGE, and cold-start clustering are
deliberately left out of this file — add them once entropy/margin/random
are validated against the orchestrator.
"""

from abc import ABC, abstractmethod
import numpy as np


class QueryStrategy(ABC):
    name: str

    @abstractmethod
    def select(self, probs: np.ndarray, k: int, **kwargs) -> np.ndarray:
        """
        probs: (n_unlabeled, N_LABELS) predicted probabilities
        k: number of samples to select
        Returns: indices into probs (local indices, 0..n_unlabeled-1)
        """
        raise NotImplementedError


class RandomStrategy(QueryStrategy):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select(self, probs: np.ndarray, k: int, **kwargs) -> np.ndarray:
        n = probs.shape[0]
        k = min(k, n)
        return self.rng.choice(n, size=k, replace=False)


class EntropyStrategy(QueryStrategy):
    """
    Per-sample entropy averaged across labels (treating each label as an
    independent binary distribution). Higher entropy = more uncertain.
    """

    name = "entropy"

    def select(self, probs: np.ndarray, k: int, **kwargs) -> np.ndarray:
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        binary_entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # (n, N_LABELS)
        sample_scores = binary_entropy.mean(axis=1)  # (n,)
        k = min(k, len(sample_scores))
        return np.argsort(sample_scores)[::-1][:k]  # highest entropy first


class MarginStrategy(QueryStrategy):
    """
    Per-label margin = |p - 0.5| (distance from the decision boundary for
    that binary classifier). Smaller margin = more uncertain. Aggregated
    across labels by taking the minimum margin (the single most uncertain
    label drives the sample's priority).
    """

    name = "margin"

    def select(self, probs: np.ndarray, k: int, **kwargs) -> np.ndarray:
        margins = np.abs(probs - 0.5)  # (n, N_LABELS)
        sample_scores = margins.min(axis=1)  # smallest margin across labels
        k = min(k, len(sample_scores))
        return np.argsort(sample_scores)[:k]  # smallest margin first (most uncertain)


# Registry for easy lookup by name (orchestrator config, eval sweeps)
STRATEGIES: dict[str, type[QueryStrategy]] = {
    "random": RandomStrategy,
    "entropy": EntropyStrategy,
    "margin": MarginStrategy,
}