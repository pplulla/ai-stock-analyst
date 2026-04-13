# ai_analyst/cache.py

from functools import lru_cache
from typing import Dict, Any
import pandas as pd

from .model import (
    train_model_for_ticker,
    run_backtest,
)

# -------------------------------
# Cached training + feature build
# -------------------------------
@lru_cache(maxsize=32)
def get_trained_result(
    ticker: str,
    price_period: str,
) -> Dict[str, Any]:
    """
    Expensive: downloads data, builds features, trains model.
    Cached per ticker + period.
    """
    return train_model_for_ticker(
        ticker=ticker,
        price_period=price_period,
    )


# -------------------------------
# Cached backtest
# -------------------------------
@lru_cache(maxsize=32)
def get_backtest_result(
    ticker: str,
    price_period: str,
) -> pd.DataFrame:
    """
    Very expensive: walk-forward backtest.
    Cached per ticker + period.
    """
    return run_backtest(
        ticker=ticker,
        price_period=price_period,
    )
