"""
Evaluation runner — Layer 5's benchmark engine.

Two responsibilities, per the spec:

1. Strategy robustness sweep: "each strategy benchmarked across all
   annotator personas, not just one noise level."
2. Policy sweep: "benchmark runner parameterized across routing
   thresholds, severity cutoffs, and personas — this is your policy
   simulation, not a separate subsystem."

Both reuse the same underlying run_single_config() — a sweep is just many
calls to it with different (strategy, persona, thresholds) combinations.
Results come back as a flat list of dicts, easy to dump to CSV/pandas for
the actual reporting/plotting step (kept out of this file — this module
only runs configs and collects raw results, plotting is presentation logic
that belongs elsewhere).
"""

from dataclasses import dataclass, asdict
import copy

import numpy as np

from app.db.pool_manager import PoolManager
from app.models.base import ModelAdapter, N_LABELS
from app.core.orchestrator import Orchestrator
from app.core.severity_router import RouterThresholds
from app.core.annotator_personas import make_annotator, PERSONAS
from app.core.query_strategies import STRATEGIES


@dataclass
class RunConfig:
    strategy_name: str
    persona_name: str
    thresholds: RouterThresholds
    batch_size: int = 40
    max_rounds: int = 5
    seed: int = 0


@dataclass
class RunSummary:
    config: RunConfig
    final_labeled_count: int
    final_f1: float | None
    mean_ece: float
    total_routed_to_human: int
    total_flagged_ambiguous: int
    total_auto_labeled: int
    final_audit_agreement_rate: float | None


def _instantiate_strategy(strategy_cls, seed: int):
    """
    Not every QueryStrategy accepts a seed kwarg (only RandomStrategy does
    currently; EntropyStrategy/MarginStrategy take no constructor args).
    Try seed first, fall back to no-arg construction rather than assuming
    a uniform constructor signature across all registered strategies.
    """
    try:
        return strategy_cls(seed=seed)
    except TypeError:
        return strategy_cls()


def run_single_config(
    config: RunConfig,
    seed_texts: list[str],
    seed_labels: np.ndarray,
    pool_texts: list[str],
    pool_true_labels: np.ndarray,
    eval_texts: list[str],
    eval_labels: np.ndarray,
    adapter_factory,
) -> RunSummary:
    """
    adapter_factory: callable() -> fresh untrained ModelAdapter, so each
    config run starts from a clean model (no leakage between sweep runs).

    seed_texts/seed_labels: pre-labeled seed set, same across all configs
    in a sweep for fair comparison.
    pool_texts/pool_true_labels: the FULL pool (seed + unlabeled), ground
    truth used only for persona simulation, exactly mirroring how
    PoolManager is normally constructed.
    eval_texts/eval_labels: fixed held-out set, never in the pool, used to
    compute final_f1 — same set across the whole sweep.
    """
    if config.strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{config.strategy_name}'. Available: {list(STRATEGIES)}")

    pool = PoolManager(pool_texts, true_labels=pool_true_labels)
    seed_ids = np.arange(len(seed_texts))
    pool.submit_labels(seed_ids, seed_labels)

    adapter = adapter_factory()
    strategy_cls = STRATEGIES[config.strategy_name]
    strategy = _instantiate_strategy(strategy_cls, config.seed)
    annotator = make_annotator(config.persona_name, seed=config.seed)

    orch = Orchestrator(
        pool=pool,
        adapter=adapter,
        query_strategy=strategy,
        annotator=annotator,
        batch_size=config.batch_size,
        max_rounds=config.max_rounds,
        thresholds=config.thresholds,
    )
    history = orch.run()

    final_f1 = None
    if len(pool.labeled_ids()) > 0:
        try:
            train_texts, train_labels = pool.get_labeled_data()
            eval_probs = adapter.predict_proba(eval_texts)
            from sklearn.metrics import f1_score
            preds = (eval_probs >= 0.5).astype(int)
            final_f1 = float(f1_score(eval_labels, preds, average="macro", zero_division=0))
        except Exception:
            final_f1 = None  # adapter may not support eval if pool ended up empty/degenerate

    mean_ece = float(np.mean([r.ece for r in history])) if history else float("nan")
    total_human = sum(r.n_routed_to_human for r in history)
    total_ambiguous = sum(r.n_flagged_ambiguous for r in history)
    total_auto = sum(r.n_auto_labeled for r in history)
    final_audit_rate = history[-1].audit_agreement_rate if history else None

    return RunSummary(
        config=config,
        final_labeled_count=len(pool.labeled_ids()),
        final_f1=final_f1,
        mean_ece=mean_ece,
        total_routed_to_human=total_human,
        total_flagged_ambiguous=total_ambiguous,
        total_auto_labeled=total_auto,
        final_audit_agreement_rate=final_audit_rate,
    )


def strategy_robustness_sweep(
    seed_texts: list[str],
    seed_labels: np.ndarray,
    pool_texts: list[str],
    pool_true_labels: np.ndarray,
    eval_texts: list[str],
    eval_labels: np.ndarray,
    adapter_factory,
    strategy_names: list[str] | None = None,
    persona_names: list[str] | None = None,
    thresholds: RouterThresholds | None = None,
    **run_kwargs,
) -> list[RunSummary]:
    """
    Cartesian product of strategy_names x persona_names. Defaults to all
    registered strategies and all 5 personas if not specified — this is
    the "benchmarked across all annotator personas, not just one noise
    level" requirement.
    """
    strategy_names = strategy_names or list(STRATEGIES.keys())
    persona_names = persona_names or list(PERSONAS.keys())
    thresholds = thresholds or RouterThresholds()

    results = []
    for strategy_name in strategy_names:
        for persona_name in persona_names:
            config = RunConfig(
                strategy_name=strategy_name,
                persona_name=persona_name,
                thresholds=copy.deepcopy(thresholds),
                **run_kwargs,
            )
            summary = run_single_config(
                config, seed_texts, seed_labels, pool_texts, pool_true_labels,
                eval_texts, eval_labels, adapter_factory,
            )
            results.append(summary)
    return results


def policy_threshold_sweep(
    seed_texts: list[str],
    seed_labels: np.ndarray,
    pool_texts: list[str],
    pool_true_labels: np.ndarray,
    eval_texts: list[str],
    eval_labels: np.ndarray,
    adapter_factory,
    confidence_thresholds: list[float],
    severity_thresholds: list[float],
    strategy_name: str = "entropy",
    persona_name: str = "strict",
    **run_kwargs,
) -> list[RunSummary]:
    """
    Cartesian product of confidence_threshold x severity_threshold, fixed
    strategy/persona, per the spec's policy simulation. Lets you see how
    routing policy choices alone (not strategy/persona) move the
    auto-label/human/ambiguous balance and exposure.
    """
    results = []
    for conf_t in confidence_thresholds:
        for sev_t in severity_thresholds:
            thresholds = RouterThresholds(confidence_threshold=conf_t, severity_threshold=sev_t)
            config = RunConfig(
                strategy_name=strategy_name,
                persona_name=persona_name,
                thresholds=thresholds,
                **run_kwargs,
            )
            summary = run_single_config(
                config, seed_texts, seed_labels, pool_texts, pool_true_labels,
                eval_texts, eval_labels, adapter_factory,
            )
            results.append(summary)
    return results


def summaries_to_records(summaries: list[RunSummary]) -> list[dict]:
    """Flatten RunSummary (incl. nested RunConfig/RouterThresholds) into flat dicts for CSV/pandas export."""
    records = []
    for s in summaries:
        record = asdict(s)
        config = record.pop("config")
        thresholds = config.pop("thresholds")
        record.update(config)
        record.update({f"threshold_{k}": v for k, v in thresholds.items()})
        records.append(record)
    return records