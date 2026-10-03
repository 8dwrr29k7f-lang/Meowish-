"""
Main 15-minute BTC prediction orchestrator.
Produces clear UP / DOWN / NEUTRAL signals with data freshness.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.critique import Critique, run_critique
from core.data_feed import DataFeed, MarketSnapshot
from core.ensemble import EnsembleResult, combine
from core.features import FeatureSet, build_features
from core.models import ALL_MODELS
from core.regime import Regime, detect_regime


# Signal thresholds — emit directional when edge is present
MIN_CONF_SIGNAL = 0.12          # was 0.28 — allow more signals
MIN_EDGE = 0.02                 # was 0.03 — |p_up - 0.5|


@dataclass
class Prediction:
    direction: str                 # UP | DOWN | NEUTRAL
    p_up: float
    p_down: float
    confidence: float
    confidence_label: str
    horizon_min: int = 15
    price: float = 0.0
    data_age_sec: float = 9999.0
    data_ts: float = 0.0
    generated_at: float = 0.0
    regime: str = "unknown"
    regime_detail: str = ""
    agreement: float = 0.0
    models_up: int = 0
    models_total: int = 0
    key_signals: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    model_detail: List[Dict[str, Any]] = field(default_factory=list)
    critique_summary: str = ""
    data_errors: List[str] = field(default_factory=list)
    usable: bool = False
    model_version: str = "v3.3-signals"
    signal: str = ""               # human-readable SIGNAL line

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.feed = DataFeed(symbol=symbol)
        self.model_version = "v3.3-signals"
        self._last: Optional[Prediction] = None

    def predict(self) -> Prediction:
        snap = self.feed.snapshot()
        fs = build_features(snap)
        regime = detect_regime(fs)

        outputs = [m.predict(fs, regime) for m in ALL_MODELS]
        ens = combine(outputs, regime_strength=regime.strength)
        critique = run_critique(snap, fs, regime, ens)

        # Apply critique multiplier — soft floor so signals still fire
        conf = ens.confidence * max(0.55, critique.confidence_multiplier)
        conf = max(0.0, min(0.90, conf))

        # Direction with tighter neutral band → more signals
        edge = abs(ens.p_up - 0.5)
        if conf < MIN_CONF_SIGNAL or edge < MIN_EDGE:
            direction = "NEUTRAL"
        elif ens.p_up >= 0.5:
            direction = "UP"
        else:
            direction = "DOWN"

        # Key signals: top reasons from models leaning with direction
        key = []
        for o in sorted(outputs, key=lambda x: abs(x.p_up - 0.5) * x.weight, reverse=True):
            for r in o.reasons[:1]:
                key.append(f"[{o.name}] {r}")
            if len(key) >= 4:
                break
        key.append(f"regime: {regime.name} ({regime.detail})")

        risks = list(critique.risk_flags) + list(critique.conflicts)
        if critique.data_issues:
            risks.extend(critique.data_issues[:2])
        if ens.notes:
            risks.extend(ens.notes)

        # Human-readable signal banner
        if direction == "NEUTRAL":
            signal = "NEUTRAL  (edge too thin — wait)"
        else:
            arrow = "▲" if direction == "UP" else "▼"
            lean = ens.models_up if direction == "UP" else (ens.models_total - ens.models_up)
            signal = (
                f"{arrow} SIGNAL {direction}  "
                f"P({direction})={max(ens.p_up, ens.p_down):.0%}  "
                f"conf={conf:.0%} ({('Low' if conf < 0.35 else 'Medium' if conf < 0.55 else 'High')})  "
                f"models {lean}/{ens.models_total} lean {direction}"
            )

        pred = Prediction(
            direction=direction,
            p_up=ens.p_up,
            p_down=ens.p_down,
            confidence=round(conf, 4),
            confidence_label="Low" if conf < 0.35 else ("High" if conf >= 0.55 else "Medium"),
            price=snap.price,
            data_age_sec=round(snap.price_age, 2),
            data_ts=snap.price_ts,
            generated_at=time.time(),
            regime=regime.name,
            regime_detail=regime.detail,
            agreement=ens.agreement,
            models_up=ens.models_up,
            models_total=ens.models_total,
            key_signals=key,
            risk_factors=risks,
            model_detail=[
                {"name": o.name, "p_up": round(o.p_up, 3), "weight": o.weight, "reasons": o.reasons}
                for o in outputs
            ],
            critique_summary=critique.summary,
            data_errors=snap.errors[:5],
            usable=snap.is_usable() and not fs.stale,
            model_version=self.model_version,
            signal=signal,
        )
        self._last = pred
        return pred

    @property
    def last(self) -> Optional[Prediction]:
        return self._last
