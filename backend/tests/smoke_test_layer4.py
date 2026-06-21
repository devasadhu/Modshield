"""
Layer 4 integration smoke test — runs the full orchestrator loop on
synthetic data. Run from backend/: python tests/smoke_test_layer4.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from app.db.pool_manager import PoolManager
from app.models.logreg_adapter import LogRegAdapter
from app.models.base import N_LABELS
from app.core.query_strategies import EntropyStrategy
from app.core.annotator_personas import make_annotator
from app.core.orchestrator import Orchestrator

rng = np.random.default_rng(0)
toxic_words = ['hate', 'stupid', 'idiot', 'kill', 'ugly']
neutral_words = ['nice', 'weather', 'today', 'coffee', 'morning']
n = 400
texts, labels = [], []
for _ in range(n):
    is_toxic = rng.random() < 0.3
    words = rng.choice(toxic_words if is_toxic else neutral_words, size=5)
    texts.append(' '.join(words))
    row = np.zeros(N_LABELS)
    if is_toxic:
        row[0] = 1
        for li in range(1, N_LABELS):
            if rng.random() < 0.15:
                row[li] = 1
    labels.append(row)
labels = np.array(labels)

pool = PoolManager(texts, true_labels=labels)
seed_ids = np.arange(30)
pool.submit_labels(seed_ids, labels[seed_ids])

adapter = LogRegAdapter()
strategy = EntropyStrategy()
annotator = make_annotator('strict', seed=1)

orch = Orchestrator(pool=pool, adapter=adapter, query_strategy=strategy, annotator=annotator, batch_size=40, max_rounds=5)
history = orch.run()

print("== Orchestrator smoke test ==")
for r in history:
    print(f"round {r.round_num}: queried={r.n_queried} auto={r.n_auto_labeled} human={r.n_routed_to_human} "
          f"ambiguous={r.n_flagged_ambiguous} ece={r.ece:.4f} thresh={r.confidence_threshold:.3f} "
          f"audit_agree={r.audit_agreement_rate}")

print("final labeled:", len(pool.labeled_ids()), "/ unlabeled:", len(pool.unlabeled_ids()))
assert len(history) > 0
print("PASS: orchestrator ran end-to-end")