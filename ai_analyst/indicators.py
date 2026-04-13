# ai_analyst/indicators.py

from typing import Dict

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)

    macd = ema_fast - ema_slow
    macd_signal = compute_ema(macd, signal)
    macd_hist = macd - macd_signal

    return pd.DataFrame(
        {
            "MACD": macd,
            "MACD_signal": macd_signal,
            "MACD_hist": macd_hist,
        }
    )


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: Date, Close, Volume
    Returns a copy with indicator columns added and initial Na rows dropped.
    """
    out = df.copy()
    close = out["Close"]

    out["RSI_14"] = compute_rsi(close, window=14)
    out["SMA_10"] = compute_sma(close, window=10)
    out["SMA_50"] = compute_sma(close, window=50)
    out["EMA_20"] = compute_ema(close, span=20)

    macd_df = compute_macd(close)
    out = pd.concat([out, macd_df], axis=1)

    out["VOL_20"] = out["Volume"].rolling(window=20, min_periods=20).mean()

    out = out.dropna().reset_index(drop=True)
    return out


INDICATOR_EXPLANATIONS: Dict[str, str] = {
    "RSI_14": (
        "Relative Strength Index over fourteen days. "
        "Values above seventy can indicate stretched upside conditions, "
        "values below thirty can indicate stretched downside conditions."
    ),
    "SMA_10": (
        "Simple moving average of the closing price over ten days. "
        "Used to capture short term trend."
    ),
    "SMA_50": (
        "Simple moving average over fifty days. "
        "Often used as a medium trend reference level."
    ),
    "EMA_20": (
        "Exponential moving average over twenty days. "
        "Gives more weight to recent prices to track short term trend."
    ),
    "MACD": (
        "Difference between fast and slow exponential averages. "
        "Positive values hint at bullish momentum, negative values at bearish momentum."
    ),
    "MACD_signal": (
        "Smoothed version of MACD. "
        "Crossovers between MACD and its signal can indicate potential turning points."
    ),
    "MACD_hist": (
        "Difference between MACD and its signal line. "
        "Shows the strength of the momentum signal."
    ),
    "VOL_20": (
        "Average traded volume over twenty days. "
        "Helps to judge whether current activity is quiet or highly active."
    ),
}
