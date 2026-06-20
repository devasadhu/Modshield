"""
Model-agnostic adapter interface.

Every classifier used in TriageLoop (DistilBERT, logistic regression baseline,
future swap-ins) must implement this interface. This is the abstraction that
lets Layer 2 (query strategies) and Layer 3 (severity router) stay agnostic
to which underlying model is running.

Multi-label setup: Jigsaw has 6 labels (toxic, severe_toxic, obscene, threat,
insult, identity_hate). predict_proba returns shape (n_samples, n_labels).
"""

from abc import ABC, abstractmethod
import numpy as np

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
N_LABELS = len(LABELS)


class ModelAdapter(ABC):
    """Common interface every model wrapper must satisfy."""

    @abstractmethod
    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        """
        Train (or fine-tune) on the given samples.

        texts: list of raw text strings, length n
        labels: np.ndarray of shape (n, N_LABELS), binary multi-label targets
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Return uncalibrated probability estimates.

        Returns: np.ndarray of shape (n, N_LABELS), values in [0, 1]
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Return a fixed-size embedding per sample, used by diversity sampling
        (core-set / k-center) and BADGE-style gradient-embedding strategies.

        Returns: np.ndarray of shape (n, embedding_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError