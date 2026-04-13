# AI Stock Analyst

A Streamlit application that analyzes stocks and indexes using technical indicators, machine learning, market regime detection, and AI-generated summaries to produce a probabilistic short-term outlook.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

Enter any stock ticker (e.g. SPY, AAPL, MSFT) and the app:

1. Downloads 2 years of daily price data via yfinance
2. Engineers technical indicators: RSI, SMA, EMA, MACD, and volume features
3. Trains a Random Forest regressor to predict the 5-day forward return
4. Runs a walk-forward backtest to measure real predictive performance over time
5. Detects the current market regime (Trending, Range-bound, High Volatility)
6. Applies regime-aware prediction clipping and confidence calibration
7. Evaluates an alert system (NO ALERT / WATCH / ACTIONABLE) using sentiment and RSI guardrails
8. Generates an AI analyst summary via the Google Gemini API

---

## Tech stack

| Layer | Tool |
|---|---|
| App framework | Streamlit |
| Data | yfinance |
| Machine learning | scikit-learn (RandomForestRegressor) |
| Sentiment | NewsAPI + TextBlob |
| AI summary | Google Gemini 2.5 Flash |
| Language | Python 3.10+ |

---

## Project structure

```
AI_Trading_App/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example      # Copy and fill with your API keys
└── ai_analyst/
    ├── pipeline.py               # Orchestration, AnalystOutput dataclass
    ├── model.py                  # Data download, feature engineering, training, backtest
    ├── indicators.py             # RSI, SMA, EMA, MACD implementations
    ├── sentiment.py              # NewsAPI fetch and TextBlob scoring
    ├── alerts.py                 # Regime-aware alert decision logic
    ├── cache.py                  # lru_cache wrappers for expensive operations
    └── summarizer.py             # Prompt builder for Gemini AI summary
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/pplulla/ai-stock-analyst.git
cd ai-stock-analyst
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Copy the example secrets file and fill in your keys:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Open `.streamlit/secrets.toml` and add:

```toml
GEMINI_API_KEY = "your-google-gemini-api-key"
NEWS_API_KEY = "your-newsapi-org-key"
```

You can get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) and a free NewsAPI key at [newsapi.org](https://newsapi.org).

### 4. Run the app

```bash
streamlit run app.py
```

---

## How the alert system works

The app does not simply output a buy or sell signal. It gates alerts through a multi-condition system:

- The predicted move must exceed a minimum threshold (0.5%)
- Confidence must meet a regime-specific floor (65% to 70% depending on regime)
- Sentiment conflict with the directional call applies a soft confidence penalty
- Overbought or oversold RSI in a Trending regime applies an additional penalty

If all conditions are met, the alert fires as WATCH or ACTIONABLE depending on the confidence margin above the required threshold.

---

## Limitations

- The model is retrained from scratch on each fresh run (no persistent model caching between sessions)
- Twitter sentiment is currently stubbed at 0.0 and does not affect results
- The walk-forward backtest can take several minutes on the first run for a new ticker
- This tool is for educational and research purposes only. It is not financial advice.

---

## Author

**Pranay Lulla**
Lead Regional Data Architect | Crawford and Company
[GitHub](https://github.com/pplulla)
