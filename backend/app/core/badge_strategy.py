"""
BADGE-style hybrid strategy.

True BADGE (Ash et al. 2020) computes gradient embeddings w.r.t. the last
layer using the model's *hypothesized* label (argmax prediction), then runs
k-means++ seeding over those gradient embeddings — a single procedure that
captures both uncertainty (gradient magnitude is larger for uncertain
points) and diversity (k-means++ spreads selections across the space).

Approximation used here (documented, not hidden): a true last-layer
gradient embedding requires backprop access per sample, which is expensive
and adapter-specific (LogReg has no "last layer" in the same sense).
Instead we build a proxy gradient embedding as:

    grad_embedding[i] = embedding[i] * uncertainty_scalar[i]

i.e. scale each sample's feature embedding by its predictive uncertainty
(mean binary entropy across labels). This preserves BADGE's core property —
points with higher uncertainty get larger-magnitude vectors, so k-means++
seeding naturally favors them while still respecting feature-space spread —
without requiring per-sample gradients. Revisit with true last-layer
gradients on DistilBERT if the proxy underperforms in the eval sweep.
"""

import numpy as np

from app.core.query_strategies import QueryStrategy


class BadgeStrategy(QueryStrategy):
    name = "badge"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select(self, probs: np.ndarray, k: int, **kwargs) -> np.ndarray:
        """
        kwargs must include `embeddings`: (n, embedding_dim) from
        ModelAdapter.embed(), aligned to the same samples as `probs`.
        """
        embeddings = kwargs.get("embeddings")
        if embeddings is None:
            raise ValueError("BadgeStrategy.select requires embeddings=... kwarg")

        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        uncertainty = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(axis=1)  # (n,)

        # proxy gradient embedding: scale features by uncertainty
        grad_embeddings = embeddings * uncertainty[:, None]

        return self._kmeans_pp_seed(grad_embeddings, k)

    def _kmeans_pp_seed(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        k-means++ seeding: pick the first center uniformly at random, then
        each subsequent center with probability proportional to its squared
        distance from the nearest already-chosen center. This is what makes
        BADGE diversity-aware rather than just "top-k by gradient norm".
        """
        n = points.shape[0]
        k = min(k, n)

        first = self.rng.integers(0, n)
        selected = [first]
        min_sq_dist = ((points - points[first]) ** 2).sum(axis=1)

        for _ in range(k - 1):
            total = min_sq_dist.sum()
            if total <= 0:
                # all remaining points coincide with a selected center;
                # fall back to uniform choice among not-yet-selected
                remaining = [i for i in range(n) if i not in selected]
                if not remaining:
                    break
                next_idx = self.rng.choice(remaining)
            else:
                probs_ = min_sq_dist / total
                next_idx = self.rng.choice(n, p=probs_)
                while next_idx in selected:
                    next_idx = self.rng.choice(n, p=probs_)

            selected.append(int(next_idx))
            new_sq_dist = ((points - points[next_idx]) ** 2).sum(axis=1)
            min_sq_dist = np.minimum(min_sq_dist, new_sq_dist)

        return np.array(selected)