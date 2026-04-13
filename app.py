# app.py

import streamlit as st
import pandas as pd
import google.generativeai as genai

from ai_analyst.pipeline import run_analyst_for_ticker
from ai_analyst.summarizer import build_ai_summary_prompt

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Stock Analyst",
    layout="wide",
)

st.title("AI Stock Analyst")
st.write(
    "This tool analyzes a stock or index using technical indicators, sentiment, "
    "machine learning, regime detection, and historical calibration to produce "
    "a probabilistic short-term outlook."
)

# --------------------------------------------------
# Secrets / API setup
# --------------------------------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    st.error("Gemini API key not found in Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_KEY)

# --------------------------------------------------
# User input
# --------------------------------------------------
symbol = st.text_input("Symbol", value="SPY").upper()
run = st.button("Run analysis")

# --------------------------------------------------
# Cached execution
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_analysis(ticker: str):
    return run_analyst_for_ticker(ticker)


# --------------------------------------------------
# Main execution
# --------------------------------------------------
if run:
    with st.spinner("Running analysis, backtest, and summary…"):
        output = cached_analysis(symbol)

        prompt = build_ai_summary_prompt(output)
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            summary = model.generate_content(prompt).text
        except Exception as exc:
            summary = f"[AI summary unavailable: {exc}]"

        # --------------------------------------------------
    # ALERT BANNER (ADD HERE)
    # --------------------------------------------------
    alert = output.alert_decision

    st.subheader("Alert status")

    level = alert["alert_level"]
    direction = alert["direction"]

    if level == "ACTIONABLE":
        st.success(f"ACTIONABLE {direction} setup")
    elif level == "WATCH":
        st.warning(f"WATCH {direction} setup")
    else:
        st.info("NO ALERT — conditions not strong enough")

    st.caption(
        f"Confidence used: {alert['confidence_used']:.0%} | "
        f"Required: {alert['required_confidence']:.0%} | "
        f"Expected move: {alert['expected_move']:.2%} | "
        f"Regime: {alert['regime']}"
    )

    with st.expander("Why this alert was (or was not) triggered"):
        st.write("Reasons")
        for r in alert["reasons"]:
            st.write(f"- {r}")

        if alert["notes"]:
            st.write("Notes")
            for n in alert["notes"]:
                st.write(f"- {n}")

    # --------------------------------------------------
    # Headline metrics
    # --------------------------------------------------
    cols = st.columns(3)
    cols[0].metric(
        "Current price",
        f"{output.current_price:.2f}",
    )
    cols[1].metric(
        "Predicted 5-day return",
        f"{output.predicted_5d_return:.2%}",
    )
    cols[2].metric(
        "Predicted price (5 days)",
        f"{output.predicted_5d_price:.2f}",
    )

    # --------------------------------------------------
    # Model quality
    # --------------------------------------------------
    st.subheader("Model quality")
    st.write(
        f"Train score: {output.model_train_score:.3f} | "
        f"Test score: {output.model_test_score:.3f}"
    )

    # --------------------------------------------------
    # Market regime
    # --------------------------------------------------
    st.subheader("Market regime")

    reg = output.market_regime
    cols = st.columns(3)
    cols[0].metric("Regime", reg["regime"])
    cols[1].metric("Trend strength", reg["trend_strength"])
    cols[2].metric("Volatility", reg["volatility"])

    # --------------------------------------------------
    # Regime suitability (critical guardrail)
    # --------------------------------------------------
    st.subheader("Regime suitability for this model")

    regime_name = reg["regime"]

    if regime_name == "High Volatility":
        suitability = "Favorable edge, higher risk"
        note = (
            "Historically strongest directional edge for this model, "
            "but price swings are larger. Position sizing matters."
        )
    elif regime_name == "Range-bound":
        suitability = "Mixed edge, smaller payoff"
        note = (
            "Hit rate can look strong, but average payoff is smaller. "
            "Be selective and mindful of costs."
        )
    elif regime_name == "Trending":
        suitability = "Caution for 5-day horizon"
        note = (
            "Historically weaker edge for this model on a five-day horizon. "
            "Short-term pullbacks and mean reversion are common."
        )
    else:
        suitability = "Unknown"
        note = "Insufficient regime data."

    cols = st.columns(2)
    cols[0].metric("Suitability", suitability)
    cols[1].write(note)

    # Historical context for current regime
    metrics = output.regime_backtest_metrics
    if regime_name in metrics:
        m = metrics[regime_name]
        st.caption(
            f"Historical performance in this regime — "
            f"Samples: {m['samples']}, "
            f"Directional accuracy: {m['directional_accuracy']:.3f}, "
            f"Mean return: {m['mean_actual_return']:.4f}, "
            f"MAE: {m['mae']:.4f}"
        )

    # --------------------------------------------------
    # Backtest summary
    # --------------------------------------------------
    st.subheader("Backtest summary (walk-forward)")

    if output.backtest_results:
        bt_df = pd.DataFrame(output.backtest_results)
        st.dataframe(bt_df.tail(15))
    else:
        st.write("Backtest data not available.")

    last_bt_date = bt_df["Date"].max()
    st.caption(
        f"Note: Backtest labels require a 5-day forward return. "
        f"Latest usable backtest date is {last_bt_date}."
    )


    # --------------------------------------------------
    # Backtest by regime
    # --------------------------------------------------
    st.subheader("Backtest performance by market regime")

    rows = []
    for regime, vals in output.regime_backtest_metrics.items():
        rows.append(
            {
                "Regime": regime,
                "Samples": vals["samples"],
                "Directional accuracy": vals["directional_accuracy"],
                "Mean actual return": vals["mean_actual_return"],
                "MAE": vals["mae"],
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.write("No regime-specific backtest metrics available.")

    # --------------------------------------------------
    # Latest indicators
    # --------------------------------------------------
    st.subheader("Latest indicators")

    feats = []
    for name, val in output.latest_features.items():
        explanation = output.indicator_explanations.get(
            name,
            "No explanation available.",
        )
        feats.append(
            {
                "Indicator": name,
                "Value": round(val, 3),
                "Explanation": explanation,
            }
        )

    st.dataframe(pd.DataFrame(feats))

    # --------------------------------------------------
    # Prediction clipping transparency
    # --------------------------------------------------
    st.subheader("Prediction magnitude control")

    clip = output.prediction_clipping
    st.write(
        f"Raw model prediction: {clip['raw_prediction']:.2%}\n\n"
        f"Clipped prediction: {clip['clipped_prediction']:.2%}\n\n"
        f"Clip limit for this regime: {clip['clip_limit']:.2%}\n\n"
        f"Was prediction clipped: {'Yes' if clip['was_clipped'] else 'No'}"
    )

    # --------------------------------------------------
    # Calibration
    # --------------------------------------------------
    st.subheader("Prediction calibration")

    cal = output.calibration
    st.write(
        f"Directional probability: {cal['directional_probability']:.0%}\n\n"
        f"Prediction percentile: {cal['prediction_percentile']:.0%}\n\n"
        f"Confidence level: {cal['confidence_level']}"
    )

    if "confidence_reasons" in cal:
        st.caption("Confidence context:")
        for r in cal["confidence_reasons"]:
            st.write(f"- {r}")

    # --------------------------------------------------
    # AI summary
    # --------------------------------------------------
    st.subheader("AI summary")
    st.write(summary)
