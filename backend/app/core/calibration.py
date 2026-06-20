"""
Post-hoc calibration: temperature scaling (primary) and Platt scaling
(secondary, for comparison). Also provides ECE (Expected Calibration Error),
tracked every round per the spec.

Multi-label note: calibration is fit per-label (one temperature, or one
Platt model, per of the N_LABELS binary classifiers) since each label has
its own confidence distribution.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.models.base import N_LABELS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class TemperatureScaler:
    """
    Fits one scalar temperature T per label by minimizing NLL on a held-out
    calibration set. Applied as: calibrated = sigmoid(logit(p) / T).
    """

    def __init__(self):
        self.temperatures: np.ndarray | None = None  # shape (N_LABELS,)

    def fit(self, probs: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> None:
        """
        probs: (n, N_LABELS) uncalibrated probabilities from the model
        labels: (n, N_LABELS) binary ground truth
        """
        from scipy.optimize import minimize

        logits = _logit(probs)
        temps = np.ones(N_LABELS)

        for i in range(N_LABELS):
            y = labels[:, i]
            z = logits[:, i]

            def nll(T):
                T = max(T[0], 1e-3)
                p = _sigmoid(z / T)
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

            res = minimize(nll, x0=[1.0], method="Nelder-Mead", options={"maxiter": max_iter})
            temps[i] = max(res.x[0], 1e-3)

        self.temperatures = temps

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.temperatures is None:
            raise RuntimeError("TemperatureScaler not fitted yet")
        logits = _logit(probs)
        return _sigmoid(logits / self.temperatures[None, :])


class PlattScaler:
    """
    Fits a 1D logistic regression per label on (logit(p) -> label) as a
    secondary calibration method, for comparison against temperature scaling.
    """

    def __init__(self):
        self.models: list[LogisticRegression] = []

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        self.models = []
        logits = _logit(probs)
        for i in range(N_LABELS):
            y = labels[:, i]
            if len(np.unique(y)) < 2:
                # Rare/absent label in this calibration slice — no signal to
                # fit a Platt model on. Store None and pass through unchanged
                # at transform time rather than crashing.
                self.models.append(None)
                continue
            lr = LogisticRegression()
            lr.fit(logits[:, i].reshape(-1, 1), y)
            self.models.append(lr)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("PlattScaler not fitted yet")
        logits = _logit(probs)
        out = np.zeros_like(probs)
        for i, lr in enumerate(self.models):
            if lr is None:
                out[:, i] = probs[:, i]  # passthrough, no calibration data was available
            else:
                out[:, i] = lr.predict_proba(logits[:, i].reshape(-1, 1))[:, 1]
        return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE averaged across all N_LABELS binary classifiers.
    Standard binning definition: |confidence - accuracy| weighted by bin size.
    """
    eces = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(N_LABELS):
        p = probs[:, i]
        y = labels[:, i]
        ece = 0.0
        n = len(p)
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
            if mask.sum() == 0:
                continue
            bin_conf = p[mask].mean()
            bin_acc = y[mask].mean()
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
        eces.append(ece)

    return float(np.mean(eces))


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """
    Returns per-bin (mean_confidence, mean_accuracy, count) averaged across
    labels, suitable for plotting a reliability diagram. Stored per round
    per the spec.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    confidences = np.zeros(n_bins)
    accuracies = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    p_flat = probs.flatten()
    y_flat = labels.flatten()

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (p_flat >= lo) & (p_flat < hi) if b < n_bins - 1 else (p_flat >= lo) & (p_flat <= hi)
        if mask.sum() == 0:
            continue
        confidences[b] = p_flat[mask].mean()
        accuracies[b] = y_flat[mask].mean()
        counts[b] = mask.sum()

    return {
        "bin_centers": bin_centers.tolist(),
        "confidences": confidences.tolist(),
        "accuracies": accuracies.tolist(),
        "counts": counts.tolist(),
    }