# ai_analyst/sentiment.py

from datetime import timedelta
from typing import List

import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob


def _analyze_sentiment(text: str) -> float:
    """
    Returns polarity in range [-1, 1]
    """
    if not text or not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity


def fetch_news_sentiment_score(
    ticker: str,
    date: pd.Timestamp,
    api_key: str,
    lookback_days: int = 3,
) -> float:
    """
    Fetch news headlines around a date and compute average sentiment.
    """

    client = NewsApiClient(api_key=api_key)

    from_date = (date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date = date.strftime("%Y-%m-%d")

    try:
        response = client.get_everything(
            q=ticker,
            language="en",
            from_param=from_date,
            to=to_date,
            sort_by="relevancy",
            page_size=20,
        )
    except Exception:
        return 0.0

    articles = response.get("articles", [])

    if not articles:
        return 0.0

    scores: List[float] = []

    for article in articles:
        headline = article.get("title", "")
        description = article.get("description", "")
        combined = f"{headline}. {description}".strip()
        scores.append(_analyze_sentiment(combined))

    return sum(scores) / len(scores) if scores else 0.0


def add_sentiment_features(
    df: pd.DataFrame,
    ticker: str,
    news_api_key: str,
) -> pd.DataFrame:
    """
    Adds:
        - news_sentiment
        - twitter_sentiment (still stubbed)
    """

    out = df.copy()

    news_scores: List[float] = []
    twitter_scores: List[float] = []

    for dt in out["Date"]:
        news_scores.append(
            fetch_news_sentiment_score(
                ticker=ticker,
                date=dt,
                api_key=news_api_key,
            )
        )
        twitter_scores.append(0.0)

    out["news_sentiment"] = news_scores
    out["twitter_sentiment"] = twitter_scores

    return out
