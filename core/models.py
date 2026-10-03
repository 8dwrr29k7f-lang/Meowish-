"""
Independent prediction models. Each returns P(UP) in [0,1] + reasons.
Tuned to produce actionable short-horizon directional signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.features import FeatureSet
from core.regime import Regime


@dataclass
class ModelOutput:
    name: str
    p_up: float
    reasons: List[str] = field(default_factory=list)
    weight: float = 1.0


def _clamp(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, x))


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


class MomentumModel:
    name = "momentum"

    def predict(self, fs: FeatureSet, regime: Regime) -> ModelOutput:
        reasons = []
        score = 0.0

        r1 = fs.ret_1m or 0.0
        r5 = fs.ret_5m or 0.0
        r15 = fs.ret_15m or 0.0
        slope = fs.slope_15m or 0.0

        if regime.name in ("trend_up", "trend_down", "breakout"):
            score += 0.70 * _sign(r5) * min(abs(r5) / 0.06, 2.0)
            score += 0.55 * _sign(slope) * min(abs(slope) / 0.02, 2.0)
            score += 0.25 * _sign(r15) * min(abs(r15) / 0.10, 1.5)
            reasons.append(f"trend continuation ({regime.name})")
        elif regime.name == "range":
            if abs(r1) > 0.08:
                score -= 0.60 * _sign(r1)
                reasons.append(f"fade 1m spike {r1:+.3f}%")
            else:
                score += 0.45 * _sign(r5) * min(abs(r5) / 0.05, 1.8)
                score += 0.35 * _sign(r15) * min(abs(r15) / 0.10, 1.8)
                reasons.append(f"range drift 5m={r5:+.3f}% 15m={r15:+.3f}%")
        else:
            score += 0.50 * _sign(r5) * min(abs(r5) / 0.06, 1.8)
            score += 0.30 * _sign(r15) * min(abs(r15) / 0.12, 1.5)
            reasons.append(f"momentum 5m={r5:+.3f}% 15m={r15:+.3f}%")

        p = _clamp(0.5 + score * 0.38)
        return ModelOutput(self.name, p, reasons or ["neutral momentum"], weight=1.25)


class OrderFlowModel:
    name = "orderflow"

    def predict(self, fs: FeatureSet, regime: Regime) -> ModelOutput:
        reasons = []
        score = 0.0
        bi = fs.book_imbalance
        ti = fs.trade_imbalance

        if abs(bi) > 0.06:
            score += 0.80 * bi
            reasons.append(f"book imbalance {bi:+.2f}")
        if abs(ti) > 0.05:
            score += 0.70 * ti
            reasons.append(f"trade aggression {ti:+.2f}")

        w = 0.70 if regime.name == "high_vol" else 1.15
        if abs(bi) < 0.01 and abs(ti) < 0.01:
            return ModelOutput(self.name, 0.5, ["no flow data"], weight=0.20)

        p = _clamp(0.5 + score * 0.42 * w)
        return ModelOutput(self.name, p, reasons or ["balanced flow"], weight=1.05 * w)


class StructureModel:
    name = "structure"

    def predict(self, fs: FeatureSet, regime: Regime) -> ModelOutput:
        reasons = []
        score = 0.0

        if fs.above_sma20_15m is True:
            score += 0.45
            reasons.append("price above SMA20(15m)")
        elif fs.above_sma20_15m is False:
            score -= 0.45
            reasons.append("price below SMA20(15m)")

        if fs.above_sma20_1h is True:
            score += 0.30
            reasons.append("above SMA20(1h)")
        elif fs.above_sma20_1h is False:
            score -= 0.30
            reasons.append("below SMA20(1h)")

        rsi = fs.rsi_14_15m
        if rsi is not None:
            if rsi >= 65:
                score -= 0.50
                reasons.append(f"RSI overbought {rsi:.0f}")
            elif rsi <= 35:
                score += 0.50
                reasons.append(f"RSI oversold {rsi:.0f}")
            else:
                score += 0.25 * ((rsi - 50) / 20)

        if regime.name == "reversal":
            r15 = fs.ret_15m or 0
            score -= 0.55 * _sign(r15)
            reasons.append("reversal regime — fade")

        p = _clamp(0.5 + score * 0.40)
        return ModelOutput(self.name, p, reasons or ["neutral structure"], weight=1.15)


class VolatilityModel:
    name = "volatility"

    def predict(self, fs: FeatureSet, regime: Regime) -> ModelOutput:
        reasons = []
        atr = fs.atr_15m_pct or 0.2
        r15 = fs.ret_15m or 0.0
        r5 = fs.ret_5m or 0.0
        volz = fs.vol_z_15m or 0.0
        score = 0.0

        if atr > 0.30 and abs(r15) > 0.12 and volz < 0.4:
            score -= 0.55 * _sign(r15)
            reasons.append("expanded range, weak volume — reversion")
        elif regime.name == "breakout" and volz > 1.0:
            score += 0.60 * _sign(r15)
            reasons.append("volume-confirmed breakout")
        elif abs(r15) > 0.08:
            score += 0.35 * _sign(r15)
            reasons.append(f"ATR={atr:.2f}% follow {r15:+.3f}%")
        elif abs(r5) > 0.05:
            score += 0.25 * _sign(r5)
            reasons.append(f"short-term follow 5m={r5:+.3f}%")
        else:
            reasons.append(f"ATR15={atr:.2f}% vol_z={volz:.1f}")

        p = _clamp(0.5 + score * 0.38)
        return ModelOutput(self.name, p, reasons, weight=0.90)


class BaselineModel:
    name = "baseline"

    def predict(self, fs: FeatureSet, regime: Regime) -> ModelOutput:
        r15 = fs.ret_15m
        r5 = fs.ret_5m
        if r15 is None and r5 is None:
            return ModelOutput(self.name, 0.5, ["no prior return"], weight=0.30)
        r = r15 if r15 is not None else (r5 or 0.0)
        strength = min(abs(r) / 0.08, 1.5)
        p = _clamp(0.5 + 0.18 * _sign(r) * strength)
        label = "15m" if r15 is not None else "5m"
        return ModelOutput(
            self.name,
            p,
            [f"prev {label} {'green' if r > 0 else 'red'} {r:+.3f}%"],
            weight=0.55,
        )


ALL_MODELS = [
    MomentumModel(),
    OrderFlowModel(),
    StructureModel(),
    VolatilityModel(),
    BaselineModel(),
]
