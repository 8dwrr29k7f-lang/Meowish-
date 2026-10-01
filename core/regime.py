"""Market regime detection from multi-TF features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.features import FeatureSet


@dataclass
class Regime:
    name: str  # trend_up | trend_down | range | high_vol | low_vol | breakout | reversal
    strength: float  # 0-1
    detail: str


def detect_regime(fs: FeatureSet) -> Regime:
    atr = fs.atr_15m_pct or 0.0
    slope = fs.slope_15m or 0.0
    slope_h = fs.slope_1h or 0.0
    ret15 = fs.ret_15m or 0.0
    volz = fs.vol_z_15m or 0.0
    rsi = fs.rsi_14_15m or 50.0

    # Volatility first
    if atr >= 0.55:
        return Regime("high_vol", min(1.0, atr / 1.0), f"ATR15={atr:.2f}% elevated")
    if atr > 0 and atr <= 0.12:
        return Regime("low_vol", min(1.0, (0.12 - atr) / 0.12), f"ATR15={atr:.2f}% compressed")

    # Breakout: volume spike + range expansion + directional move
    if volz is not None and volz > 1.8 and abs(ret15) > 0.15:
        direction = "up" if ret15 > 0 else "down"
        return Regime(
            "breakout",
            min(1.0, abs(volz) / 3),
            f"vol_z={volz:.1f} ret15={ret15:+.2f}% ({direction})",
        )

    # Reversal: RSI extreme + short-term move against higher TF slope
    if rsi >= 72 and slope_h < 0 and ret15 > 0.1:
        return Regime("reversal", 0.6, f"RSI={rsi:.0f} overbought into downtrend")
    if rsi <= 28 and slope_h > 0 and ret15 < -0.1:
        return Regime("reversal", 0.6, f"RSI={rsi:.0f} oversold into uptrend")

    # Trend
    if slope > 0.02 and slope_h > 0.01 and (fs.above_sma20_15m is True):
        return Regime("trend_up", min(1.0, abs(slope) / 0.08), f"slope15={slope:+.3f} slope1h={slope_h:+.3f}")
    if slope < -0.02 and slope_h < -0.01 and (fs.above_sma20_15m is False):
        return Regime("trend_down", min(1.0, abs(slope) / 0.08), f"slope15={slope:+.3f} slope1h={slope_h:+.3f}")

    return Regime("range", 0.5, f"slope15={slope:+.3f} ATR={atr:.2f}%")
