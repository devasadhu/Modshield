"""
Diversity sampling: core-set / k-center greedy selection over [CLS]
embeddings (or TF-IDF vectors for LogRegAdapter — same interface, see
ModelAdapter.embed).

Greedy k-center: repeatedly pick the unlabeled point that is farthest from
its nearest already-selected (or already-labeled) point. Maximizes coverage
of the embedding space rather than targeting uncertain points — this is
the complement to entropy/margin, not a replacement, which is why BADGE
combines both.
"""

import numpy as np

from query_strategies import QueryStrategy


class CoreSetStrategy(QueryStrategy):
    """
    NOTE: doesn't use predicted probabilities at all — needs embeddings.
    Like QBC, this strategy's signature doesn't fit the plain (probs, k)
    interface, so use select_with_embeddings() directly.
    """

    name = "core_set"

    def select(self, probs: np.ndarray, k: int, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "CoreSetStrategy needs embeddings, not probs — call select_with_embeddings()."
        )

    def select_with_embeddings(
        self,
        unlabeled_embeddings: np.ndarray,
        labeled_embeddings: np.ndarray | None,
        k: int,
    ) -> np.ndarray:
        """
        unlabeled_embeddings: (n_unlabeled, dim)
        labeled_embeddings: (n_labeled, dim) or None/empty for cold start
        (cold start with zero labeled points is handled by ColdStartStrategy
        instead — this assumes at least one already-labeled point, or falls
        back to picking an arbitrary first point if none exists)

        Returns: indices into unlabeled_embeddings (local indices)
        """
        n_unlabeled = unlabeled_embeddings.shape[0]
        k = min(k, n_unlabeled)

        if labeled_embeddings is None or len(labeled_embeddings) == 0:
            # No reference set yet — start from an arbitrary point (index 0)
            # and grow the core-set greedily from there.
            center_pool = unlabeled_embeddings[:1]
            selected_local = [0]
            remaining_mask = np.ones(n_unlabeled, dtype=bool)
            remaining_mask[0] = False
        else:
            center_pool = labeled_embeddings
            selected_local = []
            remaining_mask = np.ones(n_unlabeled, dtype=bool)

        # min distance from each unlabeled point to the nearest point in the
        # current center pool (labeled + newly selected)
        min_dist = self._min_dist_to_set(unlabeled_embeddings, center_pool)

        while len(selected_local) < k:
            min_dist[~remaining_mask] = -np.inf  # exclude already-picked
            next_idx = int(np.argmax(min_dist))
            if not remaining_mask[next_idx]:
                break  # exhausted candidates (shouldn't normally happen)

            selected_local.append(next_idx)
            remaining_mask[next_idx] = False

            # update min_dist with the newly added point
            new_point = unlabeled_embeddings[next_idx : next_idx + 1]
            dist_to_new = self._min_dist_to_set(unlabeled_embeddings, new_point)
            min_dist = np.minimum(min_dist, dist_to_new)

        return np.array(selected_local[: k if labeled_embeddings is not None else k])

    @staticmethod
    def _min_dist_to_set(points: np.ndarray, reference_set: np.ndarray) -> np.ndarray:
        """Euclidean distance from each row in `points` to its nearest row in `reference_set`."""
        # (n_points, 1, dim) - (1, n_ref, dim) -> (n_points, n_ref, dim)
        diffs = points[:, None, :] - reference_set[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))  # (n_points, n_ref)
        return dists.min(axis=1)