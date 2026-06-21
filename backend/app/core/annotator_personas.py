"""
Annotator persona system — simulates human labelers so the active learning
loop can be evaluated without real annotators (per spec: "no real moderator
deployment... simulated personas only").

Each persona is a config of:
- base_error_rate: probability of flipping any given label
- bias_direction: systematic tendency to over- or under-flag (not just
  random noise) — "strict" over-flags, "lenient" under-flags
- per_label_skill: some personas are better/worse at specific labels
  (e.g. "specialist" is much better at identity_hate than at obscene)
- time_decay: fatigued personas get noisier as session_position increases

Five personas per the spec: strict, lenient, noisy, fatigued, specialist.
"""

from dataclasses import dataclass, field

import numpy as np

from app.models.base import LABELS, N_LABELS


@dataclass
class PersonaConfig:
    name: str
    base_error_rate: float
    bias_direction: float          # -1 = lenient (under-flag), 0 = neutral, +1 = strict (over-flag)
    per_label_skill: dict[str, float] = field(default_factory=dict)  # label -> error_rate multiplier (1.0 = baseline)
    fatigue_rate: float = 0.0      # additional error per item within a session (0 = no fatigue)


PERSONAS: dict[str, PersonaConfig] = {
    "strict": PersonaConfig(
        name="strict",
        base_error_rate=0.08,
        bias_direction=1.0,   # tends to over-flag borderline content
        per_label_skill={},
        fatigue_rate=0.0,
    ),
    "lenient": PersonaConfig(
        name="lenient",
        base_error_rate=0.08,
        bias_direction=-1.0,  # tends to under-flag borderline content
        per_label_skill={},
        fatigue_rate=0.0,
    ),
    "noisy": PersonaConfig(
        name="noisy",
        base_error_rate=0.20,  # high random error, no systematic bias
        bias_direction=0.0,
        per_label_skill={},
        fatigue_rate=0.0,
    ),
    "fatigued": PersonaConfig(
        name="fatigued",
        base_error_rate=0.05,  # starts sharp
        bias_direction=0.0,
        per_label_skill={},
        fatigue_rate=0.01,     # but degrades steadily within a session
    ),
    "specialist": PersonaConfig(
        name="specialist",
        base_error_rate=0.10,
        bias_direction=0.0,
        # much better than baseline at identity_hate and threat (the
        # categories a domain specialist would be trained to catch),
        # worse than baseline at obscene (less attention to mild content)
        per_label_skill={"identity_hate": 0.3, "threat": 0.3, "obscene": 1.5},
        fatigue_rate=0.0,
    ),
}


class SimulatedAnnotator:
    def __init__(self, persona: PersonaConfig, seed: int | None = None):
        self.persona = persona
        self.rng = np.random.default_rng(seed)
        self.session_position = 0  # increments per item labeled, drives fatigue

    def _effective_error_rate(self, label: str) -> float:
        skill_multiplier = self.persona.per_label_skill.get(label, 1.0)
        fatigue_addon = self.persona.fatigue_rate * self.session_position
        rate = self.persona.base_error_rate * skill_multiplier + fatigue_addon
        return float(np.clip(rate, 0.0, 0.95))

    def label(self, true_labels: np.ndarray) -> np.ndarray:
        """
        true_labels: (N_LABELS,) ground-truth binary vector for one sample.
        Returns: (N_LABELS,) simulated human label, with errors applied
        per-label according to this persona's error rate, bias, skill, and
        current fatigue level. Advances session_position by 1.
        """
        result = true_labels.copy().astype(int)

        for idx, label_name in enumerate(LABELS):
            error_rate = self._effective_error_rate(label_name)
            if self.rng.random() >= error_rate:
                continue  # no error on this label

            true_val = true_labels[idx]
            if self.persona.bias_direction > 0:
                # strict bias: errors skew toward flipping 0 -> 1 (over-flagging)
                flip_prob = 0.5 + 0.5 * self.persona.bias_direction
            elif self.persona.bias_direction < 0:
                # lenient bias: errors skew toward flipping 1 -> 0 (under-flagging)
                flip_prob = 0.5 + 0.5 * self.persona.bias_direction
            else:
                flip_prob = 0.5

            # decide whether THIS particular error instance flips toward 1 or toward 0,
            # but only actually changes the value if it differs from true_val
            wants_flip_to_one = self.rng.random() < flip_prob
            if true_val == 0 and wants_flip_to_one:
                result[idx] = 1
            elif true_val == 1 and not wants_flip_to_one:
                result[idx] = 0
            # else: the "error" coincidentally matches truth, no visible change

        self.session_position += 1
        return result

    def label_batch(self, true_labels_batch: np.ndarray) -> np.ndarray:
        """true_labels_batch: (n, N_LABELS). Returns: (n, N_LABELS)."""
        return np.stack([self.label(row) for row in true_labels_batch])

    def reset_session(self) -> None:
        self.session_position = 0


def make_annotator(persona_name: str, seed: int | None = None) -> SimulatedAnnotator:
    if persona_name not in PERSONAS:
        raise ValueError(f"Unknown persona '{persona_name}'. Available: {list(PERSONAS)}")
    return SimulatedAnnotator(PERSONAS[persona_name], seed=seed)