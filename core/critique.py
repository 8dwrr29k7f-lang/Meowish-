"""
Pre-prediction self-critique. Surfaces conflicts and data issues.
Does not invent confidence — only adjusts downward when risks found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.data_feed import MarketSnapshot
from core.ensemble import EnsembleResult
from core.features import FeatureSet
from core.regime import Regime


@dataclass
class Critique:
    support_up: List[str] = field(default_factory=list)
    support_down: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    data_issues: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    confidence_multiplier: float = 1.0
    summary: str = ""


def run_critique(
    snap: MarketSnapshot,
    fs: FeatureSet,
    regime: Regime,
    ens: EnsembleResult,
) -> Critique:
    c = Critique()

    # Data reliability
    if snap.price_age > 8:
        c.data_issues.append(f"price data age {snap.price_age:.1f}s — stale risk")
        c.confidence_multiplier *= 0.7
    if snap.book_stale:
        c.data_issues.append("order book stale or missing")
        c.confidence_multiplier *= 0.9
    if snap.price_divergence_bps > 25:
        c.data_issues.append(
            f"venue divergence {snap.price_divergence_bps:.0f} bps"
        )
        c.confidence_multiplier *= 0.85
    if snap.errors:
        c.data_issues.extend(snap.errors[:3])

    # Evidence up / down from features
    if (fs.ret_5m or 0) > 0.05:
        c.support_up.append(f"5m return {fs.ret_5m:+.3f}%")
    if (fs.ret_5m or 0) < -0.05:
        c.support_down.append(f"5m return {fs.ret_5m:+.3f}%")
    if (fs.slope_15m or 0) > 0.02:
        c.support_up.append(f"15m slope {fs.slope_15m:+.3f}")
    if (fs.slope_15m or 0) < -0.02:
        c.support_down.append(f"15m slope {fs.slope_15m:+.3f}")
    if fs.book_imbalance > 0.2:
        c.support_up.append(f"bid-heavy book {fs.book_imbalance:+.2f}")
    if fs.book_imbalance < -0.2:
        c.support_down.append(f"ask-heavy book {fs.book_imbalance:+.2f}")
    if fs.trade_imbalance > 0.2:
        c.support_up.append(f"buy aggression {fs.trade_imbalance:+.2f}")
    if fs.trade_imbalance < -0.2:
        c.support_down.append(f"sell aggression {fs.trade_imbalance:+.2f}")
    if fs.above_sma20_15m is True:
        c.support_up.append("above SMA20 15m")
    if fs.above_sma20_15m is False:
        c.support_down.append("below SMA20 15m")

    # Conflicts (lighter penalties so signals still fire)
    if c.support_up and c.support_down:
        c.conflicts.append("both UP and DOWN evidence present")
        c.confidence_multiplier *= 0.88
    if ens.agreement < 0.5:
        c.conflicts.append(f"model agreement only {ens.agreement:.0%}")
        c.confidence_multiplier *= 0.90

    # Regime risks
    if regime.name == "high_vol":
        c.risk_flags.append("high volatility — short-horizon noise elevated")
        c.confidence_multiplier *= 0.85
    if regime.name == "reversal":
        c.risk_flags.append("possible reversal regime — trend signals unreliable")
        c.confidence_multiplier *= 0.88
    if (fs.atr_15m_pct or 0) < 0.08:
        c.risk_flags.append("very low volatility — edge may be smaller than spread")

    # Near-even ensemble
    if abs(ens.p_up - 0.5) < 0.02:
        c.risk_flags.append("ensemble probability near 50/50")
        c.confidence_multiplier *= 0.80

    c.confidence_multiplier = max(0.55, min(1.0, c.confidence_multiplier))

    parts = []
    if c.support_up:
        parts.append(f"UP evidence: {len(c.support_up)}")
    if c.support_down:
        parts.append(f"DOWN evidence: {len(c.support_down)}")
    if c.conflicts:
        parts.append(f"conflicts: {len(c.conflicts)}")
    if c.data_issues:
        parts.append(f"data issues: {len(c.data_issues)}")
    c.summary = "; ".join(parts) if parts else "no strong evidence either side"
    return c
