"""
Feature engineering from multi-timeframe OHLCV + book + trades.
All features are causal (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.data_feed import Candle, MarketSnapshot


def _returns(closes: List[float], n: int) -> Optional[float]:
    if len(closes) < n + 1 or closes[-n - 1] <= 0:
        return None
    return (closes[-1] / closes[-n - 1] - 1.0) * 100.0


def _sma(xs: List[float], n: int) -> Optional[float]:
    if len(xs) < n:
        return None
    return sum(xs[-n:]) / n


def _ema(xs: List[float], n: int) -> Optional[float]:
    if len(xs) < n:
        return None
    k = 2 / (n + 1)
    e = xs[-n]
    for x in xs[-n + 1 :]:
        e = x * k + e * (1 - k)
    return e


def _rsi(closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    gains = losses = 0.0
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100 - 100 / (1 + rs)


def _atr(candles: List[Candle], n: int = 14) -> Optional[float]:
    if len(candles) < n + 1:
        return None
    trs = []
    for i in range(-n, 0):
        c = candles[i]
        prev = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - prev), abs(c.low - prev))
        trs.append(tr)
    return sum(trs) / n


def _vol_z(volumes: List[float], n: int = 20) -> Optional[float]:
    if len(volumes) < n:
        return None
    window = volumes[-n:]
    mu = sum(window) / n
    var = sum((v - mu) ** 2 for v in window) / n
    sd = var ** 0.5
    if sd < 1e-12:
        return 0.0
    return (volumes[-1] - mu) / sd


def _slope(closes: List[float], n: int) -> Optional[float]:
    """Linear slope of last n closes, normalized by price (%)."""
    if len(closes) < n or closes[-1] <= 0:
        return None
    xs = list(range(n))
    ys = closes[-n:]
    xbar = (n - 1) / 2
    ybar = sum(ys) / n
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    if den == 0:
        return 0.0
    slope = num / den
    return slope / closes[-1] * 100.0


@dataclass
class FeatureSet:
    price: float = 0.0
    ret_1m: Optional[float] = None
    ret_3m: Optional[float] = None
    ret_5m: Optional[float] = None
    ret_15m: Optional[float] = None
    ret_30m: Optional[float] = None
    ret_1h: Optional[float] = None
    slope_15m: Optional[float] = None
    slope_1h: Optional[float] = None
    rsi_14_15m: Optional[float] = None
    rsi_14_1h: Optional[float] = None
    atr_15m_pct: Optional[float] = None
    atr_1h_pct: Optional[float] = None
    range_15m_pct: Optional[float] = None
    vol_z_15m: Optional[float] = None
    vol_z_1h: Optional[float] = None
    above_sma20_15m: Optional[bool] = None
    above_sma20_1h: Optional[bool] = None
    dist_sma20_15m_pct: Optional[float] = None
    book_imbalance: float = 0.0
    trade_imbalance: float = 0.0
    spread_bps: float = 0.0
    funding_rate: Optional[float] = None
    data_age_sec: float = 9999.0
    stale: bool = True
    regime_hint: str = "unknown"
    raw: Dict = field(default_factory=dict)


def build_features(snap: MarketSnapshot) -> FeatureSet:
    fs = FeatureSet(price=snap.price)
    fs.book_imbalance = snap.imbalance
    fs.trade_imbalance = snap.trade_imbalance
    fs.funding_rate = snap.funding_rate
    fs.data_age_sec = snap.price_age
    fs.stale = snap.price_age > 10 or not snap.is_usable()

    if snap.bid > 0 and snap.ask > 0 and snap.price > 0:
        fs.spread_bps = (snap.ask - snap.bid) / snap.price * 10000

    def closes(iv: str) -> List[float]:
        return [c.close for c in snap.klines.get(iv, [])]

    def vols(iv: str) -> List[float]:
        return [c.volume for c in snap.klines.get(iv, [])]

    c1 = closes("1m")
    c5 = closes("5m")
    c15 = closes("15m")
    c30 = closes("30m")
    c1h = closes("1h")

    fs.ret_1m = _returns(c1, 1) if c1 else None
    fs.ret_3m = _returns(c1, 3) if c1 else None
    fs.ret_5m = _returns(c5, 1) if c5 else None
    fs.ret_15m = _returns(c15, 1) if c15 else None
    fs.ret_30m = _returns(c30, 1) if c30 else None
    fs.ret_1h = _returns(c1h, 1) if c1h else None

    fs.slope_15m = _slope(c15, 8) if c15 else None
    fs.slope_1h = _slope(c1h, 12) if c1h else None
    fs.rsi_14_15m = _rsi(c15, 14) if c15 else None
    fs.rsi_14_1h = _rsi(c1h, 14) if c1h else None

    candles15 = snap.klines.get("15m") or []
    candles1h = snap.klines.get("1h") or []
    atr15 = _atr(candles15, 14)
    atr1h = _atr(candles1h, 14)
    if atr15 and snap.price > 0:
        fs.atr_15m_pct = atr15 / snap.price * 100
    if atr1h and snap.price > 0:
        fs.atr_1h_pct = atr1h / snap.price * 100
    if candles15 and snap.price > 0:
        last = candles15[-1]
        fs.range_15m_pct = (last.high - last.low) / snap.price * 100

    fs.vol_z_15m = _vol_z(vols("15m"), 20)
    fs.vol_z_1h = _vol_z(vols("1h"), 20)

    sma20_15 = _sma(c15, 20)
    sma20_1h = _sma(c1h, 20)
    if sma20_15 and snap.price > 0:
        fs.above_sma20_15m = snap.price > sma20_15
        fs.dist_sma20_15m_pct = (snap.price / sma20_15 - 1) * 100
    if sma20_1h:
        fs.above_sma20_1h = snap.price > sma20_1h

    fs.raw = {
        "ret_1m": fs.ret_1m,
        "ret_5m": fs.ret_5m,
        "ret_15m": fs.ret_15m,
        "slope_15m": fs.slope_15m,
        "rsi_15m": fs.rsi_14_15m,
        "atr_15m_pct": fs.atr_15m_pct,
        "book_imb": fs.book_imbalance,
        "trade_imb": fs.trade_imbalance,
        "vol_z_15m": fs.vol_z_15m,
    }
    return fs
