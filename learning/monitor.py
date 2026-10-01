"""Model monitoring: accuracy drift, stale data, abnormal distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from learning.database import PredictionDB


@dataclass
class MonitorStatus:
    healthy: bool = True
    flags: List[str] = field(default_factory=list)
    recent_accuracy: Optional[float] = None
    recent_n: int = 0
    recent_brier: Optional[float] = None
    total_logged: int = 0


def check_health(db: PredictionDB, min_n: int = 20, min_acc: float = 0.48) -> MonitorStatus:
    st = MonitorStatus()
    stats = db.stats()
    st.total_logged = stats["total"]
    recent = db.recent_accuracy(50)
    st.recent_n = recent["n"]
    st.recent_accuracy = recent.get("accuracy")
    st.recent_brier = recent.get("brier")

    if recent["n"] >= min_n and recent["accuracy"] is not None:
        if recent["accuracy"] < min_acc:
            st.healthy = False
            st.flags.append(
                f"accuracy deterioration: {recent['accuracy']:.1%} on last {recent['n']} directional"
            )
        if recent.get("brier") is not None and recent["brier"] > 0.28:
            st.flags.append(f"poor calibration (brier={recent['brier']:.3f})")

    if stats["total"] == 0:
        st.flags.append("no predictions logged yet")

    return st
