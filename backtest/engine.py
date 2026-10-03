"""Walk-forward backtest stub — full engine optional at runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BacktestResult:
    n: int = 0
    accuracy: float = 0.0
    f1: float = 0.0
    brier: float = 0.0
    baseline_prev_acc: float = 0.0
    neutral_rate: float = 0.0
    by_regime: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def run_walk_forward(bars_15m: int = 400) -> BacktestResult:
    """Lightweight placeholder so boot never blocks on missing history."""
    return BacktestResult(
        n=0,
        accuracy=0.0,
        f1=0.0,
        brier=0.0,
        baseline_prev_acc=0.5,
        neutral_rate=0.0,
        by_regime={},
        notes=["backtest deferred — live mode only on Railway"],
    )
