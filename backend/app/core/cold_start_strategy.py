"""
Cold-start strategy: pick the initial seed set before any labels exist, so
round 0 isn't just random sampling.

Uses k-means clustering over embeddings (e.g. TF-IDF vectors for
LogRegAdapter, or a frozen pretrained DistilBERT's [CLS] embeddings before
any fine-tuning) and selects the point closest to each cluster centroid.
This gives a seed set that already spans the dataset's natural structure,
rather than hoping random sampling happens to do that by luck.

No sklearn KMeans import — implemented directly with a simple Lloyd's
algorithm so this file has no hidden dependency surprises and is easy to
swap out later (e.g. for k-means++ init shared with BadgeStrategy).
"""

import numpy as np


class ColdStartStrategy:
    name = "cold_start"

    def __init__(self, seed: int | None = None, max_iter: int = 50):
        self.rng = np.random.default_rng(seed)
        self.max_iter = max_iter

    def select(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        """
        embeddings: (n, dim) — entire unlabeled pool's embeddings
        k: seed set size (= number of clusters)

        Returns: indices into embeddings, one per cluster, closest to each
        centroid (so selected points are real samples, not synthetic
        centroids).
        """
        n = embeddings.shape[0]
        k = min(k, n)

        centroids = self._kmeans(embeddings, k)

        # for each centroid, pick the closest real sample not yet selected
        selected = []
        used = set()
        for c in centroids:
            dists = np.sqrt(((embeddings - c) ** 2).sum(axis=1))
            order = np.argsort(dists)
            for idx in order:
                if int(idx) not in used:
                    selected.append(int(idx))
                    used.add(int(idx))
                    break

        return np.array(selected)

    def _kmeans(self, points: np.ndarray, k: int) -> np.ndarray:
        n = points.shape[0]
        init_idx = self.rng.choice(n, size=k, replace=False)
        centroids = points[init_idx].copy()

        for _ in range(self.max_iter):
            dists = np.sqrt(((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))  # (n, k)
            assignments = dists.argmin(axis=1)

            new_centroids = centroids.copy()
            for c in range(k):
                members = points[assignments == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                # if a cluster lost all members, keep its old centroid
                # rather than letting it collapse to NaN

            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids

        return centroids