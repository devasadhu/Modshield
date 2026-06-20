"""
Layer 1 smoke test. Synthetic data, LogRegAdapter only (fast).
Run: python -m backend.tests.smoke_test_layer1   (from repo root)
or:  cd backend && python -m tests.smoke_test_layer1

Validates: pool manager state transitions, adapter fit/predict_proba/embed,
calibration fit/transform, ECE computation. If this passes, the abstraction
holds and you can move to real Jigsaw data + Layer 2 query strategies.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.pool_manager import PoolManager, SampleState
from app.models.logreg_adapter import LogRegAdapter
from app.models.base import N_LABELS
from app.core.calibration import TemperatureScaler, PlattScaler, expected_calibration_error


def make_synthetic_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    toxic_words = ["hate", "stupid", "idiot", "kill", "ugly"]
    neutral_words = ["nice", "weather", "today", "coffee", "morning"]

    texts, labels = [], []
    for _ in range(n):
        is_toxic = rng.random() < 0.3
        words = rng.choice(toxic_words if is_toxic else neutral_words, size=5)
        texts.append(" ".join(words))
        row = np.zeros(N_LABELS)
        if is_toxic:
            row[0] = 1  # toxic
            # spread activity across the other labels so calibration has
            # signal for each of the 6 classes, not just toxic/severe_toxic
            for label_idx in range(1, N_LABELS):
                if rng.random() < 0.15:
                    row[label_idx] = 1
        labels.append(row)
    return texts, np.array(labels)


def main():
    print("== Layer 1 smoke test ==")
    texts, labels = make_synthetic_data(n=500)
    pool = PoolManager(texts, true_labels=labels)

    # simulate: label first 300 as a "seed set"
    seed_ids = np.arange(300)
    pool.submit_labels(seed_ids, labels[seed_ids])
    assert (pool.df.loc[pool.df.sample_id.isin(seed_ids), "state"] == SampleState.LABELED.value).all()
    print(f"[ok] pool manager state transitions, {len(pool.labeled_ids())} labeled")

    train_texts, train_labels = pool.get_labeled_data()
    adapter = LogRegAdapter()
    adapter.fit(train_texts, train_labels)
    print("[ok] adapter.fit")

    unlabeled_ids = pool.unlabeled_ids()
    eval_texts = pool.get_texts(unlabeled_ids)
    probs = adapter.predict_proba(eval_texts)
    assert probs.shape == (len(unlabeled_ids), N_LABELS)
    print(f"[ok] adapter.predict_proba, shape={probs.shape}")

    embs = adapter.embed(eval_texts)
    assert embs.shape[0] == len(unlabeled_ids)
    print(f"[ok] adapter.embed, shape={embs.shape}")

    # calibration: fit on a held-out slice of labeled data, evaluate on the rest
    cal_texts, cal_labels = train_texts[:100], train_labels[:100]
    cal_probs = adapter.predict_proba(cal_texts)

    temp_scaler = TemperatureScaler()
    temp_scaler.fit(cal_probs, cal_labels)
    calibrated = temp_scaler.transform(cal_probs)
    ece_before = expected_calibration_error(cal_probs, cal_labels)
    ece_after = expected_calibration_error(calibrated, cal_labels)
    print(f"[ok] temperature scaling: ECE before={ece_before:.4f} after={ece_after:.4f}")

    platt = PlattScaler()
    platt.fit(cal_probs, cal_labels)
    platt_calibrated = platt.transform(cal_probs)
    ece_platt = expected_calibration_error(platt_calibrated, cal_labels)
    print(f"[ok] platt scaling: ECE={ece_platt:.4f}")

    audit_df = pool.export_audit_log()
    print(f"[ok] audit log export, {len(audit_df)} entries")

    print("\nAll Layer 1 checks passed.")


if __name__ == "__main__":
    main()