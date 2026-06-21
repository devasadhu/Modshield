"""
Stopping criterion for the active learning loop.

Per the spec: "halt when marginal accuracy gain per N new labels drops
below a threshold over a moving window."

Decoupled from Orchestrator itself (orchestrator.py currently just runs
max_rounds) so the stopping logic can be swapped/tuned independently and
tested in isolation. Wire into Orchestrator.run() by checking
.should_stop(history) each round instead of only checking max_rounds.

Needs an accuracy signal per round — uses held-out F1 (computed by the
caller, e.g. on a fixed validation slice) rather than ECE, since ECE
measures calibration quality, not classification accuracy. These are
different things and shouldn't be conflated for stopping decisions.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class StoppingConfig:
    window: int = 3              # how many recent rounds to average the marginal gain over
    min_gain_threshold: float = 0.005   # stop if average marginal gain/label falls below this
    min_rounds_before_stopping: int = 3  # never stop before this many rounds, even if gain looks flat early


class StoppingCriterion:
    def __init__(self, config: StoppingConfig | None = None):
        self.config = config or StoppingConfig()
        self.accuracy_history: list[float] = []      # one accuracy value per round
        self.labels_added_history: list[int] = []     # labels added that round
        self._marginal_gains: deque[float] = deque(maxlen=self.config.window)

    def record_round(self, accuracy: float, n_labels_added: int) -> None:
        """
        Call once per round with that round's held-out accuracy (e.g. F1)
        and how many new labels were added this round.
        """
        if self.accuracy_history:
            prev_accuracy = self.accuracy_history[-1]
            gain = accuracy - prev_accuracy
            marginal_gain_per_label = gain / max(n_labels_added, 1)
            self._marginal_gains.append(marginal_gain_per_label)

        self.accuracy_history.append(accuracy)
        self.labels_added_history.append(n_labels_added)

    def should_stop(self) -> bool:
        """
        Returns True if the loop should halt: average marginal gain per
        label over the last `window` rounds has dropped below threshold,
        AND at least `min_rounds_before_stopping` rounds have run.
        """
        n_rounds = len(self.accuracy_history)
        if n_rounds < self.config.min_rounds_before_stopping:
            return False

        if len(self._marginal_gains) < self.config.window:
            return False  # not enough history yet to trust the average

        avg_marginal_gain = sum(self._marginal_gains) / len(self._marginal_gains)
        return avg_marginal_gain < self.config.min_gain_threshold

    def summary(self) -> dict:
        return {
            "n_rounds": len(self.accuracy_history),
            "accuracy_history": list(self.accuracy_history),
            "labels_added_history": list(self.labels_added_history),
            "recent_marginal_gains": list(self._marginal_gains),
            "stopped": self.should_stop(),
        }