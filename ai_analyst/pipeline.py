# ai_analyst/pipeline.py

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import pandas as pd

from .model import (
    FEATURE_COLUMNS,
    calibrate_prediction,
    detect_market_regime,
    clip_prediction_by_regime,
    compute_regime_backtest_metrics,
)
from .cache import (
    get_trained_result,
    get_backtest_result,
)
from .indicators import INDICATOR_EXPLANATIONS
from .alerts import decide_alert


@dataclass
class AnalystOutput:
    ticker: str
    as_of_date: str
    current_price: float
    predicted_5d_return: float
    predicted_5d_price: float
    model_train_score: float
    model_test_score: float
    latest_features: Dict[str, float]
    latest_sentiment: Dict[str, float]
    indicator_explanations: Dict[str, str]
    recent_bars: List[Dict[str, Any]]
    top_feature_drivers: Dict[str, float]
    backtest_results: List[Dict[str, Any]]
    calibration: Dict[str, Any]
    market_regime: Dict[str, Any]
    prediction_clipping: Dict[str, float]
    regime_backtest_metrics: Dict[str, Dict[str, float]]
    alert_decision: Dict[str, Any]


def run_analyst_for_ticker(
    ticker: str,
    price_period: str = "2y",
) -> AnalystOutput:

    # -------------------------------------------------
    # Cached training + feature engineering
    # -------------------------------------------------
    result = get_trained_result(
        ticker=ticker,
        price_period=price_period,
    )

    df: pd.DataFrame = result["data"]
    pipe = result["pipeline"]

    # -------------------------------------------------
    # Market regime detection
    # -------------------------------------------------
    market_regime = detect_market_regime(df)

    # -------------------------------------------------
    # Feature importance
    # -------------------------------------------------
    raw_importance = result["feature_importance"]
    total = sum(raw_importance.values())
    normalized = (
        {k: v / total for k, v in raw_importance.items()}
        if total > 0
        else {}
    )

    top_feature_drivers = dict(
        sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
    )

    # -------------------------------------------------
    # Latest prediction
    # -------------------------------------------------
    latest_row = df.iloc[-1]
    X_latest = latest_row[FEATURE_COLUMNS].to_frame().T

    raw_pred = float(pipe.predict(X_latest)[0])

    clip_info = clip_prediction_by_regime(
        predicted_return=raw_pred,
        market_regime=market_regime,
    )

    predicted_return = clip_info["clipped_prediction"]

    current_price = float(latest_row["Close"])
    predicted_price = current_price * (1 + predicted_return)

    # -------------------------------------------------
    # Latest feature values & sentiment
    # -------------------------------------------------
    latest_features = {
        c: float(latest_row[c])
        for c in FEATURE_COLUMNS
    }

    latest_sentiment = {
        "news_sentiment": float(latest_row["news_sentiment"]),
        "twitter_sentiment": float(latest_row["twitter_sentiment"]),
    }

    # -------------------------------------------------
    # Recent bars (for charts)
    # -------------------------------------------------
    history_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    recent_df = df[history_cols].tail(20).copy()
    recent_df["Date"] = recent_df["Date"].astype(str)
    recent_bars = recent_df.to_dict(orient="records")

    # -------------------------------------------------
    # Cached walk-forward backtest
    # -------------------------------------------------
    backtest_df = get_backtest_result(
        ticker=ticker,
        price_period=price_period,
    )

    backtest_results = backtest_df.to_dict(orient="records")

    regime_backtest_metrics = compute_regime_backtest_metrics(
        backtest_df
    )

    # -------------------------------------------------
    # Prediction calibration
    # -------------------------------------------------
    calibration = calibrate_prediction(
        backtest_df=backtest_df,
        current_pred=predicted_return,
        market_regime=market_regime,
    )

    # -------------------------------------------------
    # Build output object
    # -------------------------------------------------
    output = AnalystOutput(
        ticker=ticker,
        as_of_date=str(latest_row["Date"]),
        current_price=current_price,
        predicted_5d_return=predicted_return,
        predicted_5d_price=predicted_price,
        model_train_score=float(result["train_score"]),
        model_test_score=float(result["test_score"]),
        latest_features=latest_features,
        latest_sentiment=latest_sentiment,
        indicator_explanations=INDICATOR_EXPLANATIONS,
        recent_bars=recent_bars,
        top_feature_drivers=top_feature_drivers,
        backtest_results=backtest_results,
        calibration=calibration,
        market_regime=market_regime,
        prediction_clipping=clip_info,
        regime_backtest_metrics=regime_backtest_metrics,
        alert_decision={},  # filled next
    )

    # -------------------------------------------------
    # Alert gating decision
    # -------------------------------------------------
    alert = decide_alert(output)
    output.alert_decision = alert.to_dict()

    return output


def to_streamlit_payload(
    output: AnalystOutput,
) -> Dict[str, Any]:
    return asdict(output)
