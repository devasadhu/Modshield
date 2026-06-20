"""
Logistic regression baseline adapter.

Build and validate the whole loop against this BEFORE touching DistilBERT.
It's fast (seconds, not GPU-minutes), so use it to debug the orchestrator,
query strategies, and calibration pipeline end-to-end first.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

from app.models.base import ModelAdapter, N_LABELS


class LogRegAdapter(ModelAdapter):
    def __init__(self, max_features: int = 20_000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self._fitted = False

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        X = self.vectorizer.fit_transform(texts) if not self._fitted else self.vectorizer.transform(texts)
        if not self._fitted:
            self.clf.fit(X, labels)
            self._fitted = True
        else:
            # OneVsRestClassifier doesn't support incremental fit cleanly;
            # for active learning rounds we refit on the full accumulated pool.
            self.clf.fit(X, labels)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Adapter not fitted yet")
        X = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(X)
        return np.asarray(probs)

    def embed(self, texts: list[str]) -> np.ndarray:
        # TF-IDF vector itself serves as the embedding for diversity sampling.
        # Dense conversion — fine at the dataset sizes used here.
        X = self.vectorizer.transform(texts)
        return X.toarray()

    def save(self, path: str) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf, "fitted": self._fitted}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.vectorizer = obj["vectorizer"]
        self.clf = obj["clf"]
        self._fitted = obj["fitted"]