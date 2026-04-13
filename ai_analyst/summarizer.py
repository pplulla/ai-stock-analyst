from __future__ import annotations

from typing import Dict

from .pipeline import AnalystOutput
from .indicators import INDICATOR_EXPLANATIONS


def build_indicator_block(latest_features: Dict[str, float]) -> str:
    lines = []
    for name, value in latest_features.items():
        explanation = INDICATOR_EXPLANATIONS.get(name, "No explanation available yet.")
        lines.append(f"{name}: {value:.3f}  Explanation: {explanation}")
    return "\n".join(lines)


def build_feature_importance_block(top_drivers: Dict[str, float]) -> str:
    lines = []
    for name, weight in top_drivers.items():
        pct = weight * 100
        explanation = INDICATOR_EXPLANATIONS.get(name, "No explanation available yet.")
        lines.append(f"{name} ({pct:.1f}%): {explanation}")
    return "\n".join(lines)


def build_sentiment_block(latest_sentiment: Dict[str, float]) -> str:
    lines = []
    for src, val in latest_sentiment.items():
        lines.append(f"{src}: {val:.3f}")
    return "\n".join(lines)


def build_calibration_block(cal: Dict[str, object]) -> str:
    prob = float(cal["directional_probability"])
    pct = float(cal["prediction_percentile"])
    level = str(cal["confidence_level"])
    reasons = cal.get("confidence_reasons", [])
    reasons_text = "\n".join([f"{r}" for r in reasons]) if isinstance(reasons, list) else str(reasons)

    return (
        f"Directional probability: {prob:.0%}\n"
        f"Prediction percentile: {pct:.0%}\n"
        f"Confidence level: {level}\n"
        f"Confidence reasons:\n{reasons_text}"
    )


def build_regime_block(regime: Dict[str, object]) -> str:
    return (
        f"Market regime: {regime['regime']}\n"
        f"Trend strength: {regime['trend_strength']}\n"
        f"Volatility: {regime['volatility']}"
    )


def build_clipping_block(clip: Dict[str, object]) -> str:
    raw_pred = float(clip["raw_prediction"])
    clipped_pred = float(clip["clipped_prediction"])
    clip_limit = float(clip["clip_limit"])
    was = "Yes" if bool(clip["was_clipped"]) else "No"
    return (
        f"Raw model output: {raw_pred:.2%}\n"
        f"Clipped prediction: {clipped_pred:.2%}\n"
        f"Clip limit for regime: {clip_limit:.2%}\n"
        f"Was prediction clipped: {was}"
    )


def build_ai_summary_prompt(output: AnalystOutput) -> str:
    feature_importance_block = build_feature_importance_block(output.top_feature_drivers)
    indicators_block = build_indicator_block(output.latest_features)
    sentiment_block = build_sentiment_block(output.latest_sentiment)
    calibration_block = build_calibration_block(output.calibration)
    regime_block = build_regime_block(output.market_regime)
    clipping_block = build_clipping_block(output.prediction_clipping)

    if abs(output.predicted_5d_return) < 0.02:
        direction_hint = "slight"
    else:
        direction_hint = "meaningful"

    prompt = f"""
You are an equity analyst who writes short, clear notes for an active trader.

Context
Stock ticker: {output.ticker}
As of date: {output.as_of_date}
Current price: {output.current_price:.2f}

Model forecast
Predicted five day return (clipped): {output.predicted_5d_return:.3%}
Predicted price in five trading days: {output.predicted_5d_price:.2f}

Model quality
Train score: {output.model_train_score:.3f}
Test score: {output.model_test_score:.3f}

Prediction calibration
{calibration_block}

Market regime context
{regime_block}

Prediction magnitude control
The raw output is clipped to keep expected moves realistic for the current regime.
{clipping_block}

Key model drivers
These features contributed most to the forecast.
{feature_importance_block}

Technical indicators snapshot
{indicators_block}

News and social sentiment
Positive above zero, negative below zero
{sentiment_block}

Task
Write a brief, human friendly summary for a retail trader.

Rules
1. Start with a one line directional view.
2. Use probabilistic language and avoid point forecast certainty.
3. Respect the clipped prediction magnitude. Do not exaggerate the move size. Describe the expected move as {direction_hint}.
4. Interpret regimes using these constraints for this specific model and five day horizon:
   a. High Volatility tends to show stronger directional edge but requires stricter risk control language.
   b. Range bound can show high hit rate but usually smaller payoff. Do not oversell it.
   c. Trending has historically shown weaker edge for this model on a five day horizon. Reduce confidence and mention possible pullbacks.
5. Base reasoning primarily on the key model drivers. Mention technical indicators only if they support or conflict with those drivers.
6. Comment briefly on whether sentiment supports or conflicts with the setup.
7. End with a clear risk reminder and suggest responsible position sizing.

Avoid mentioning prompts, training data, or internal mechanics. Keep tone factual and balanced.
"""
    return prompt.strip()
