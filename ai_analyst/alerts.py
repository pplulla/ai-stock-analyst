# ai_analyst/alerts.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

#from .pipeline import AnalystOutput


@dataclass
class AlertDecision:
    alert_level: str  # NO ALERT | WATCH | ACTIONABLE
    direction: str    # BULLISH | BEARISH | FLAT
    confidence_used: float
    required_confidence: float
    expected_move: float
    regime: str
    reasons: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _direction_from_return(predicted_return: float, eps: float = 1e-9) -> str:
    if predicted_return > eps:
        return "BULLISH"
    if predicted_return < -eps:
        return "BEARISH"
    return "FLAT"


def _safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def decide_alert(
    output: AnalystOutput,
    *,
    min_abs_move: float = 0.005,
    base_min_confidence: float = 0.60,
    action_margin: float = 0.05,
) -> AlertDecision:
    """
    Decide whether an alert should fire.

    Inputs come from AnalystOutput, not raw data.
    Gating is based on:
      - regime-specific confidence threshold
      - minimum expected move (after clipping)
      - basic sentiment conflict downgrade (soft)
      - trend extreme guardrail in Trending regime (soft)
    """

    regime = str(output.market_regime.get("regime", "Unknown"))
    predicted_return = float(output.predicted_5d_return)
    expected_move = abs(predicted_return)

    cal = output.calibration or {}
    directional_prob = float(_safe_get(cal, "directional_probability", 0.5))

    direction = _direction_from_return(predicted_return)

    reasons: List[str] = []
    notes: List[str] = []

    # ---------------------------------------------------------
    # Hard gate 1: minimum expected move
    # ---------------------------------------------------------
    if expected_move < float(min_abs_move):
        reasons.append(
            f"Expected move {expected_move:.2%} is below minimum threshold {min_abs_move:.2%}."
        )
        return AlertDecision(
            alert_level="NO ALERT",
            direction=direction,
            confidence_used=directional_prob,
            required_confidence=1.0,
            expected_move=expected_move,
            regime=regime,
            reasons=reasons,
            notes=notes,
        )

    reasons.append(
        f"Expected move {expected_move:.2%} meets minimum threshold {min_abs_move:.2%}."
    )

    # ---------------------------------------------------------
    # Regime-specific confidence threshold
    # These are aligned to your observed regime metrics.
    # ---------------------------------------------------------
    if regime == "High Volatility":
        required_conf = 0.65
    elif regime == "Range-bound":
        required_conf = 0.70
    elif regime == "Trending":
        required_conf = 0.68
    else:
        required_conf = 0.70

    # Also enforce a global floor
    required_conf = max(required_conf, float(base_min_confidence))

    # ---------------------------------------------------------
    # Soft adjustments (do not block immediately)
    # ---------------------------------------------------------
    confidence_used = directional_prob

    # Sentiment conflict soft penalty
    news_sent = float(output.latest_sentiment.get("news_sentiment", 0.0))
    tw_sent = float(output.latest_sentiment.get("twitter_sentiment", 0.0))
    combined_sent = (news_sent + tw_sent) / 2.0

    if direction == "BULLISH" and combined_sent < -0.10:
        confidence_used -= 0.03
        notes.append("Sentiment conflicts with bullish call. Reduced confidence slightly.")
    elif direction == "BEARISH" and combined_sent > 0.10:
        confidence_used -= 0.03
        notes.append("Sentiment conflicts with bearish call. Reduced confidence slightly.")
    else:
        notes.append("Sentiment is neutral or aligned with the directional call.")

    # Trending extreme guardrail
    if regime == "Trending":
        rsi = float(output.latest_features.get("RSI_14", 50.0))
        if direction == "BULLISH" and rsi >= 70.0:
            confidence_used -= 0.05
            notes.append("Trending plus overbought RSI suggests pullback risk. Reduced confidence.")
        elif direction == "BEARISH" and rsi <= 30.0:
            confidence_used -= 0.05
            notes.append("Trending plus oversold RSI suggests snapback risk. Reduced confidence.")
        else:
            notes.append("Trending regime without RSI extreme. No additional penalty.")

    # Keep confidence in [0, 1]
    confidence_used = max(0.0, min(1.0, confidence_used))

    # ---------------------------------------------------------
    # Determine alert level
    # ---------------------------------------------------------
    if confidence_used < required_conf:
        reasons.append(
            f"Confidence {confidence_used:.0%} is below required {required_conf:.0%} for regime {regime}."
        )
        return AlertDecision(
            alert_level="NO ALERT",
            direction=direction,
            confidence_used=confidence_used,
            required_confidence=required_conf,
            expected_move=expected_move,
            regime=regime,
            reasons=reasons,
            notes=notes,
        )

    reasons.append(
        f"Confidence {confidence_used:.0%} meets required {required_conf:.0%} for regime {regime}."
    )

    # WATCH vs ACTIONABLE
    if confidence_used >= required_conf + float(action_margin):
        alert_level = "ACTIONABLE"
        reasons.append(
            f"Confidence exceeds required threshold by at least {action_margin:.0%}."
        )
    else:
        alert_level = "WATCH"
        reasons.append(
            f"Confidence is above threshold but within {action_margin:.0%} margin."
        )

    # Range-bound caution note
    if regime == "Range-bound":
        notes.append("Range-bound regimes can have smaller payoff even when accurate. Avoid overtrading.")

    # High volatility risk note
    if regime == "High Volatility":
        notes.append("High volatility regime. Consider smaller sizing and wider stops if trading.")

    return AlertDecision(
        alert_level=alert_level,
        direction=direction,
        confidence_used=confidence_used,
        required_confidence=required_conf,
        expected_move=expected_move,
        regime=regime,
        reasons=reasons,
        notes=notes,
    )
