"""
Pool manager: tracks sample state through the active learning loop.

States: unlabeled -> in_review -> labeled (or back to unlabeled if a label
gets contested/invalidated). Full audit trail kept for compliance/reporting
(Layer 5 audit log export).

v0: in-memory + pandas, backed by SQLite/Postgres later via app/db.
No DB wiring yet — this is the data structure the DB layer will persist.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import pandas as pd

from app.models.base import N_LABELS


class SampleState(str, Enum):
    UNLABELED = "unlabeled"
    IN_REVIEW = "in_review"
    LABELED = "labeled"
    AUTO_LABELED = "auto_labeled"


@dataclass
class AuditEntry:
    sample_id: int
    event: str  # e.g. "queried", "routed_to_human", "auto_labeled", "labeled", "audit_sampled"
    round_num: int
    timestamp: str
    extra: dict = field(default_factory=dict)


class PoolManager:
    def __init__(self, texts: list[str], true_labels: np.ndarray | None = None):
        """
        true_labels is optional and only used for simulation (persona-based
        simulated annotators draw from this instead of a real human).
        In a real deployment this would not exist at pool-init time.
        """
        n = len(texts)
        self.df = pd.DataFrame({
            "sample_id": np.arange(n),
            "text": texts,
            "state": [SampleState.UNLABELED.value] * n,
        })
        self._true_labels = true_labels  # shape (n, N_LABELS), simulation only

        self.labels = np.full((n, N_LABELS), np.nan)  # assigned labels, filled as they come in
        self.audit_log: list[AuditEntry] = []
        self.round_num = 0

    # -- queries --

    def unlabeled_ids(self) -> np.ndarray:
        return self.df.loc[self.df.state == SampleState.UNLABELED.value, "sample_id"].to_numpy()

    def labeled_ids(self) -> np.ndarray:
        return self.df.loc[self.df.state.isin([SampleState.LABELED.value, SampleState.AUTO_LABELED.value]), "sample_id"].to_numpy()

    def get_texts(self, ids: np.ndarray) -> list[str]:
        return self.df.set_index("sample_id").loc[ids, "text"].tolist()

    def get_labeled_data(self) -> tuple[list[str], np.ndarray]:
        ids = self.labeled_ids()
        texts = self.get_texts(ids)
        return texts, self.labels[ids]

    # -- mutations --

    def mark_in_review(self, ids: np.ndarray, event: str = "routed_to_human") -> None:
        self.df.loc[self.df.sample_id.isin(ids), "state"] = SampleState.IN_REVIEW.value
        for sid in ids:
            self._log(int(sid), event)

    def submit_labels(self, ids: np.ndarray, labels: np.ndarray, auto: bool = False) -> None:
        """
        labels: shape (len(ids), N_LABELS)
        auto=True marks these as auto-labeled (model decision, not human)
        """
        self.labels[ids] = labels
        state = (SampleState.AUTO_LABELED if auto else SampleState.LABELED).value
        self.df.loc[self.df.sample_id.isin(ids), "state"] = state
        event = "auto_labeled" if auto else "labeled"
        for sid in ids:
            self._log(int(sid), event)

    def audit_sample(self, ids: np.ndarray) -> None:
        """Re-route a subset of auto-labeled items to a human for silent-degradation checks."""
        self.df.loc[self.df.sample_id.isin(ids), "state"] = SampleState.IN_REVIEW.value
        for sid in ids:
            self._log(int(sid), "audit_sampled")

    def next_round(self) -> None:
        self.round_num += 1

    def _log(self, sample_id: int, event: str, **extra) -> None:
        self.audit_log.append(AuditEntry(
            sample_id=sample_id,
            event=event,
            round_num=self.round_num,
            timestamp=datetime.now(timezone.utc).isoformat(),
            extra=extra,
        ))

    def export_audit_log(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"sample_id": e.sample_id, "event": e.event, "round": e.round_num, "timestamp": e.timestamp, **e.extra}
            for e in self.audit_log
        ])

    # -- simulation helper --

    def get_true_labels(self, ids: np.ndarray) -> np.ndarray:
        if self._true_labels is None:
            raise RuntimeError("No ground truth available (not a simulation pool)")
        return self._true_labels[ids]