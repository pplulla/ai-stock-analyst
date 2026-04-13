from typing import Any, Dict, Tuple

import os
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import add_technical_indicators
from .sentiment import add_sentiment_features

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "RSI_14",
    "SMA_10",
    "SMA_50",
    "EMA_20",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "VOL_20",
    "news_sentiment",
    "twitter_sentiment",
]

TARGET_COLUMN = "forward_return_5d"


def download_price_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)

    df = df.dropna().reset_index(drop=True)
    return df


def add_forward_return(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["future_close"] = out["Close"].shift(-forward_days)
    out[TARGET_COLUMN] = (out["future_close"] - out["Close"]) / out["Close"]
    out = out.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return out


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_model_pipeline() -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    return pipe


def detect_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
    recent = df.tail(60).copy()

    trend_ma = recent["Close"].rolling(50).mean()
    if pd.isna(trend_ma.iloc[-1]):
        trend_strength = 0.0
    else:
        trend_strength = (recent["Close"].iloc[-1] - trend_ma.iloc[-1]) / recent["Close"].iloc[-1]

    returns = recent["Close"].pct_change().dropna()
    volatility = float(returns.std()) if len(returns) > 1 else 0.0

    if abs(trend_strength) > 0.02 and volatility < 0.015:
        regime = "Trending"
    elif volatility > 0.02:
        regime = "High Volatility"
    else:
        regime = "Range-bound"

    return {
        "regime": regime,
        "trend_strength": round(float(trend_strength), 4),
        "volatility": round(float(volatility), 4),
    }


def clip_prediction_by_regime(
    predicted_return: float,
    market_regime: Dict[str, Any],
) -> Dict[str, Any]:
    regime = market_regime.get("regime", "Unknown")

    if regime == "High Volatility":
        clip_limit = 0.03
    elif regime == "Range-bound":
        clip_limit = 0.015
    elif regime == "Trending":
        clip_limit = 0.012
    else:
        clip_limit = 0.015

    pr = float(predicted_return)
    clipped_prediction = max(-clip_limit, min(clip_limit, pr))

    return {
        "raw_prediction": pr,
        "clipped_prediction": float(clipped_prediction),
        "clip_limit": float(clip_limit),
        "was_clipped": abs(pr) > clip_limit,
    }


def train_model_for_ticker(
    ticker: str,
    price_period: str = "2y",
) -> Dict[str, Any]:
    raw = download_price_data(ticker, period=price_period)
    with_target = add_forward_return(raw, forward_days=5)
    with_indicators = add_technical_indicators(with_target)

    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        raise RuntimeError("NEWS_API_KEY not set in environment or Streamlit secrets.")

    full_df = add_sentiment_features(
        with_indicators,
        ticker,
        news_api_key=news_api_key,
    )

    X, y = build_feature_matrix(full_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    pipe = build_model_pipeline()
    pipe.fit(X_train, y_train)

    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)

    model = pipe.named_steps["model"]
    feature_importance = {
        FEATURE_COLUMNS[i]: float(model.feature_importances_[i])
        for i in range(len(FEATURE_COLUMNS))
    }

    return {
        "ticker": ticker,
        "data": full_df,
        "pipeline": pipe,
        "train_score": float(train_score),
        "test_score": float(test_score),
        "feature_importance": feature_importance,
    }


def run_backtest(
    ticker: str,
    price_period: str = "2y",
    test_window: int = 120,
) -> pd.DataFrame:
    raw = download_price_data(ticker, period=price_period)
    with_target = add_forward_return(raw, forward_days=5)
    with_indicators = add_technical_indicators(with_target)

    with_indicators["news_sentiment"] = 0.0
    with_indicators["twitter_sentiment"] = 0.0

    records = []
    n = len(with_indicators)
    if n <= test_window:
        return pd.DataFrame(records)

    for i in range(test_window, n):
        train_df = with_indicators.iloc[:i].copy()
        test_row = with_indicators.iloc[i]

        X_train, y_train = build_feature_matrix(train_df)

        pipe = build_model_pipeline()
        pipe.fit(X_train, y_train)

        X_test = test_row[FEATURE_COLUMNS].to_frame().T
        pred_return = float(pipe.predict(X_test)[0])
        actual_return = float(test_row[TARGET_COLUMN])

        regime_info = detect_market_regime(with_indicators.iloc[: i + 1])

        records.append(
            {
                "Date": test_row["Date"],
                "Predicted_Return": pred_return,
                "Actual_Return": actual_return,
                "Regime": regime_info["regime"],
            }
        )

    return pd.DataFrame(records)


def compute_regime_backtest_metrics(
    backtest_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    if backtest_df.empty or "Regime" not in backtest_df.columns:
        return metrics

    for regime, grp in backtest_df.groupby("Regime"):
        if grp.empty:
            continue

        directional_accuracy = ((grp["Predicted_Return"] > 0) == (grp["Actual_Return"] > 0)).mean()
        mean_actual_return = float(grp["Actual_Return"].mean())
        mae = float((grp["Predicted_Return"] - grp["Actual_Return"]).abs().mean())

        metrics[str(regime)] = {
            "samples": int(len(grp)),
            "directional_accuracy": round(float(directional_accuracy), 3),
            "mean_actual_return": round(mean_actual_return, 4),
            "mae": round(mae, 4),
        }

    return metrics


def calibrate_prediction(
    backtest_df: pd.DataFrame,
    current_pred: float,
    market_regime: Dict[str, Any],
    regime_metrics: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    preds = backtest_df["Predicted_Return"]
    cp = float(current_pred)

    percentile = float((preds < cp).mean())

    bullish_mask = preds > 0
    bearish_mask = preds < 0

    bullish_hit_rate = (
        (backtest_df.loc[bullish_mask, "Actual_Return"] > 0).mean()
        if bullish_mask.any()
        else 0.5
    )

    bearish_hit_rate = (
        (backtest_df.loc[bearish_mask, "Actual_Return"] < 0).mean()
        if bearish_mask.any()
        else 0.5
    )

    base_prob = float(bullish_hit_rate) if cp > 0 else float(bearish_hit_rate)

    regime = market_regime.get("regime", "Unknown")

    if regime == "High Volatility":
        adjusted_prob = base_prob + 0.05
    elif regime == "Range-bound":
        adjusted_prob = base_prob + 0.00
    elif regime == "Trending":
        adjusted_prob = base_prob - 0.07
    else:
        adjusted_prob = base_prob

    adjusted_prob = max(0.50, min(0.75, float(adjusted_prob)))

    if adjusted_prob >= 0.65:
        confidence = "High"
    elif adjusted_prob >= 0.55:
        confidence = "Moderate"
    else:
        confidence = "Low"

    reasons: list[str] = []

    if regime == "Trending":
        reasons.append("Trending regime historically shows weaker edge for this 5 day horizon in your backtest.")
    elif regime == "Range-bound":
        reasons.append("Range bound regimes can show high hit rate but usually have smaller payoff and are sensitive to costs.")
    elif regime == "High Volatility":
        reasons.append("High volatility regimes historically show stronger directional edge but require tighter risk control.")

    if regime_metrics and regime in regime_metrics:
        r = regime_metrics[regime]
        reasons.append(
            f"Backtest in this regime: accuracy {r['directional_accuracy']:.3f}, mean return {r['mean_actual_return']:.4f}."
        )

    return {
        "prediction_percentile": round(percentile, 3),
        "directional_probability": round(adjusted_prob, 3),
        "confidence_level": confidence,
        "base_probability": round(base_prob, 3),
        "regime_adjustment": round(adjusted_prob - base_prob, 3),
        "confidence_reasons": reasons,
    }
