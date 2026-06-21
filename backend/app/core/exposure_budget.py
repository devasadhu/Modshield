"""
Exposure budget tracking.

Per the spec: "per-annotator cumulative severity-weighted exposure; stop
routing severe content to an annotator once a session threshold is hit;
batch and pace delivery rather than dumping severe items consecutively."

This is the piece that actually protects reviewers, not just a bookkeeping
metric — the router can produce a perfectly good ROUTE_TO_HUMAN decision
for a sample, but THIS module decides which specific annotator (if any)
that sample can currently be assigned to, and forces a cooldown/pacing
pattern so severe items don't land back-to-back for the same person.
"""

from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone

import numpy as np


@dataclass
class AnnotatorExposureState:
    annotator_id: str
    session_exposure: float = 0.0          # cumulative severity-weighted exposure this session
    items_this_session: int = 0
    recent_severities: deque = field(default_factory=lambda: deque(maxlen=10))
    last_assigned_at: str | None = None


class ExposureBudgetTracker:
    def __init__(
        self,
        session_budget: float = 10.0,
        consecutive_severe_limit: int = 2,
        severe_threshold: float = 0.6,
        cooldown_after_consecutive: int = 3,
    ):
        """
        session_budget: max cumulative severity-weighted exposure allowed
        per annotator per session before they're excluded from further
        severe-content assignment (not excluded from ALL assignment —
        low-severity items can still go to them).

        consecutive_severe_limit: max number of severe items (severity >=
        severe_threshold) allowed back-to-back before forcing a pause —
        this is the "pace delivery rather than dumping severe items
        consecutively" requirement.

        cooldown_after_consecutive: how many subsequent items must be
        non-severe (or skipped) before consecutive-severe assignment is
        allowed again for that annotator.
        """
        self.session_budget = session_budget
        self.consecutive_severe_limit = consecutive_severe_limit
        self.severe_threshold = severe_threshold
        self.cooldown_after_consecutive = cooldown_after_consecutive
        self.states: dict[str, AnnotatorExposureState] = {}
        self._consecutive_severe_count: dict[str, int] = {}
        self._cooldown_remaining: dict[str, int] = {}

    def _get_state(self, annotator_id: str) -> AnnotatorExposureState:
        if annotator_id not in self.states:
            self.states[annotator_id] = AnnotatorExposureState(annotator_id=annotator_id)
            self._consecutive_severe_count[annotator_id] = 0
            self._cooldown_remaining[annotator_id] = 0
        return self.states[annotator_id]

    def can_assign(self, annotator_id: str, severity: float) -> bool:
        """
        Returns whether this annotator can currently receive an item of
        this severity, given session budget + consecutive-severe pacing.
        Low-severity items always pass the consecutive-severe check (only
        budget matters for them); severe items additionally check pacing.
        """
        state = self._get_state(annotator_id)
        is_severe = severity >= self.severe_threshold

        # session budget check applies to everyone
        if state.session_exposure >= self.session_budget:
            return False

        if is_severe:
            if self._cooldown_remaining[annotator_id] > 0:
                return False
            if self._consecutive_severe_count[annotator_id] >= self.consecutive_severe_limit:
                return False

        return True

    def record_assignment(self, annotator_id: str, severity: float) -> None:
        """Call after actually assigning an item to update tracking state."""
        state = self._get_state(annotator_id)
        state.session_exposure += severity
        state.items_this_session += 1
        state.recent_severities.append(severity)
        state.last_assigned_at = datetime.now(timezone.utc).isoformat()

        is_severe = severity >= self.severe_threshold
        if is_severe:
            self._consecutive_severe_count[annotator_id] += 1
            if self._consecutive_severe_count[annotator_id] >= self.consecutive_severe_limit:
                self._cooldown_remaining[annotator_id] = self.cooldown_after_consecutive
        else:
            # a non-severe item breaks the consecutive-severe streak
            self._consecutive_severe_count[annotator_id] = 0
            if self._cooldown_remaining[annotator_id] > 0:
                self._cooldown_remaining[annotator_id] -= 1

    def assign_batch(
        self,
        item_ids: np.ndarray,
        severities: np.ndarray,
        annotator_ids: list[str],
    ) -> dict[str, list]:
        """
        Greedy batch assignment: for each item (in order), assign to the
        first available annotator (by can_assign) among annotator_ids,
        round-robin starting point so load isn't always biased toward the
        first annotator in the list.

        Returns: {annotator_id: [item_ids assigned]}, plus a special key
        "unassigned" for items nobody could currently take (all budgets/
        cooldowns exhausted) — caller should hold these for a later round
        rather than force an assignment that violates exposure protection.
        """
        assignments: dict[str, list] = {a: [] for a in annotator_ids}
        assignments["unassigned"] = []

        n_annotators = len(annotator_ids)
        if n_annotators == 0:
            assignments["unassigned"] = list(item_ids)
            return assignments

        start = 0
        for item_id, severity in zip(item_ids, severities):
            assigned = False
            for offset in range(n_annotators):
                idx = (start + offset) % n_annotators
                aid = annotator_ids[idx]
                if self.can_assign(aid, severity):
                    self.record_assignment(aid, severity)
                    assignments[aid].append(item_id)
                    start = (idx + 1) % n_annotators
                    assigned = True
                    break
            if not assigned:
                assignments["unassigned"].append(item_id)

        return assignments

    def reset_session(self, annotator_id: str) -> None:
        """Call at the start of a new session (e.g. new day) for this annotator."""
        if annotator_id in self.states:
            self.states[annotator_id].session_exposure = 0.0
            self.states[annotator_id].items_this_session = 0
            self._consecutive_severe_count[annotator_id] = 0
            self._cooldown_remaining[annotator_id] = 0