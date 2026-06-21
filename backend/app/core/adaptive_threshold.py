"""
Adaptive auto-label threshold.

Per the spec: "tightens when recent ECE is high (model poorly calibrated,
be conservative), loosens when well-calibrated."

Wraps RouterThresholds.confidence_threshold specifically — the other three
thresholds (epistemic/aleatoric/severity) are left untouched since ECE is
a calibration-quality signal, and only confidence_threshold is downstream
of calibration directly. Adjusting epistemic/aleatoric thresholds based on
ECE would conflate two different uncertainty signals; out of scope here.

Design: maintains a rolling window of recent per-round ECE values and
adjusts confidence_threshold proportionally, clamped to a safe range so it
never auto-labels everything (loosened too far) or routes everything to
humans (tightened too far, defeating the point of auto-labeling).
"""

from collections import deque

from app.core.severity_router import RouterThresholds


class AdaptiveThresholdController:
    def __init__(
        self,
        base_confidence_threshold: float = 0.85,
        min_threshold: float = 0.70,
        max_threshold: float = 0.97,
        ece_window: int = 5,
        ece_good: float = 0.03,   # ECE at/below this -> well-calibrated, loosen
        ece_bad: float = 0.10,    # ECE at/above this -> poorly calibrated, tighten
        step: float = 0.02,       # how much to adjust per round
    ):
        self.base = base_confidence_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.ece_window = ece_window
        self.ece_good = ece_good
        self.ece_bad = ece_bad
        self.step = step

        self.recent_ece: deque[float] = deque(maxlen=ece_window)
        self.current_threshold = base_confidence_threshold

    def update(self, latest_ece: float) -> float:
        """
        Call once per round with that round's ECE (post-calibration, from
        Layer 1's expected_calibration_error). Returns the updated
        confidence_threshold to use for the next round's routing.
        """
        self.recent_ece.append(latest_ece)
        avg_ece = sum(self.recent_ece) / len(self.recent_ece)

        if avg_ece >= self.ece_bad:
            self.current_threshold += self.step  # tighten: demand more confidence
        elif avg_ece <= self.ece_good:
            self.current_threshold -= self.step  # loosen: trust the model more
        # else: in between -> hold steady, no adjustment

        self.current_threshold = max(self.min_threshold, min(self.max_threshold, self.current_threshold))
        return self.current_threshold

    def apply(self, thresholds: RouterThresholds) -> RouterThresholds:
        """
        Returns a new RouterThresholds with confidence_threshold replaced
        by the current adaptive value; other fields copied unchanged.
        """
        return RouterThresholds(
            confidence_threshold=self.current_threshold,
            epistemic_threshold=thresholds.epistemic_threshold,
            aleatoric_threshold=thresholds.aleatoric_threshold,
            severity_threshold=thresholds.severity_threshold,
        )