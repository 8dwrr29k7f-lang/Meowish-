"""
Ensemble: weighted average of independent models + agreement.
Produces clear directional confidence when models lean together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.models import ModelOutput


@dataclass
class EnsembleResult:
    p_up: float
    p_down: float
    confidence: float          # 0-1 certainty of direction
    confidence_label: str      # Low | Medium | High
    agreement: float           # fraction of models on the same side of 0.5
    models_up: int
    models_total: int
    model_outputs: List[ModelOutput] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _label(conf: float) -> str:
    if conf >= 0.55:
        return "High"
    if conf >= 0.35:
        return "Medium"
    return "Low"


def combine(outputs: List[ModelOutput], regime_strength: float = 0.5) -> EnsembleResult:
    if not outputs:
        return EnsembleResult(
            0.5, 0.5, 0.0, "Low", 0.0, 0, 0, [], ["no model outputs"]
        )

    # Weighted probability
    num = den = 0.0
    for o in outputs:
        num += o.p_up * o.weight
        den += o.weight
    p_up = num / den if den > 0 else 0.5
    p_down = 1.0 - p_up

    # Agreement: how many models on same side as ensemble
    side_up = p_up >= 0.5
    agree = sum(1 for o in outputs if (o.p_up >= 0.5) == side_up)
    agreement = agree / len(outputs)

    # Edge magnitude
    edge = abs(p_up - 0.5)

    # Confidence: stronger mapping so real edges become actionable signals
    dispersion = sum(abs(o.p_up - p_up) for o in outputs) / len(outputs)
    conf = edge * 3.0 * agreement * (1.0 - min(dispersion, 0.30) / 0.30 * 0.30)
    conf *= 0.90 + 0.15 * regime_strength
    conf = max(0.0, min(0.90, conf))

    notes = []
    if agreement < 0.5:
        notes.append("models disagree — confidence reduced")
        conf *= 0.80
    if edge < 0.025:
        notes.append("edge near zero")
        conf *= 0.70

    return EnsembleResult(
        p_up=round(p_up, 4),
        p_down=round(p_down, 4),
        confidence=round(conf, 4),
        confidence_label=_label(conf),
        agreement=round(agreement, 3),
        models_up=sum(1 for o in outputs if o.p_up >= 0.5),
        models_total=len(outputs),
        model_outputs=outputs,
        notes=notes,
    )
