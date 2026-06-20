# ModShield (TriageLoop)

Active-learning content triage system that separates two objectives most
moderation tooling conflates: **labeling efficiency** (reach target accuracy
with fewest human-reviewed samples) and **exposure reduction** (of samples
that do need review, avoid sending severe content to a human when
high-confidence automation could handle it). Built on the Jigsaw Toxic
Comment dataset (multi-label: toxic, severe_toxic, obscene, threat, insult,
identity_hate).

Status: Layers 1–2 done, Layer 3 in progress. Architecture and structure
will keep changing — this README is a placeholder, not a final spec.

## Layers

- **Layer 1 — Data, Model & Calibration** (`backend/app/models/`,
  `backend/app/core/calibration.py`, `backend/app/db/pool_manager.py`):
  model-agnostic adapter interface (DistilBERT + logistic regression
  baseline), temperature/Platt scaling, ECE tracking, pool state + audit log.
- **Layer 2 — Query Strategy Engine** (`backend/app/core/`): entropy,
  margin, random baselines; MC-dropout uncertainty decomposition
  (epistemic/aleatoric); query-by-committee; core-set/k-center diversity
  sampling; BADGE-style hybrid; cold-start clustering; cost-weighted scoring.
- **Layer 3 — Severity-Aware Router** (core contribution, in progress):
  calibrated confidence + uncertainty type + severity estimate decides
  auto-label / audit-sample / route-to-human / flag-as-ambiguous.
- **Layer 4 — Labeling Interface & Orchestrator** (not started)
- **Layer 5 — Evaluation & Reporting** (not started)

## Running the Layer 1 smoke test

```
cd backend
pip install -r requirements.txt
python tests/smoke_test_layer1.py
```

## Stack

Python 3.11, PyTorch + HuggingFace Transformers, scikit-learn, FastAPI
(planned), PostgreSQL (planned), React/Vite (planned).