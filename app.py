from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ml_utils import align_feature_columns, build_single_input

st.set_page_config(page_title="AI4ALL Stock Risk Predictor", page_icon="📉", layout="wide")

MODELS_DIR = Path("models")
EXAMPLES = {
    "Strong earnings headline": {
        "ticker": "AAPL",
        "headline": "Apple beats earnings estimates as iPhone demand remains strong",
        "today_price": 210.50,
        "one_day_ago": 208.10,
        "two_days_ago": 205.80,
    },
    "Negative legal headline": {
        "ticker": "TSLA",
        "headline": "Tesla faces new lawsuit as margins fall and deliveries miss forecasts",
        "today_price": 244.30,
        "one_day_ago": 249.10,
        "two_days_ago": 252.40,
    },
    "Neutral analyst update": {
        "ticker": "MSFT",
        "headline": "Microsoft holds steady after analysts reiterate long-term cloud outlook",
        "today_price": 468.20,
        "one_day_ago": 467.80,
        "two_days_ago": 466.50,
    },
}


@st.cache_resource
def load_artifacts():
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
    metrics = {}
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return rf, lr, feature_columns, metrics


if not MODELS_DIR.exists():
    st.error("Model files are missing. Run: `python train_models.py`")
    st.stop()

rf_model, lr_model, feature_columns, metrics = load_artifacts()

st.title("AI4ALL Stock Drop Predictor")
st.write("Enter a headline and recent prices to predict whether the stock may decrease the next day.")

top_cols = st.columns(4)
with top_cols[0]:
    example_name = st.selectbox("Example", list(EXAMPLES.keys()))
with top_cols[1]:
    chosen_model = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
with top_cols[2]:
    st.metric("Random Forest accuracy", f"{metrics.get('random_forest_accuracy', 0.0):.2%}" if metrics else "n/a")
with top_cols[3]:
    st.metric(
        "Logistic Regression accuracy",
        f"{metrics.get('logistic_regression_accuracy', 0.0):.2%}" if metrics else "n/a",
    )

example = EXAMPLES[example_name]

left_col, right_col = st.columns([1.2, 0.8], gap="large")
with left_col:
    st.subheader("Inputs")
    with st.form("prediction_form"):
        ticker = st.text_input("Stock ticker", value=example["ticker"]).upper().strip()
        headline = st.text_area("News headline", value=example["headline"], height=140).strip()

        col1, col2, col3 = st.columns(3)
        with col1:
            today_price = st.number_input(
                "Today's price", min_value=0.01, value=float(example["today_price"]), step=0.01
            )
        with col2:
            one_day_ago = st.number_input(
                "1 day ago price", min_value=0.01, value=float(example["one_day_ago"]), step=0.01
            )
        with col3:
            two_days_ago = st.number_input(
                "2 days ago price", min_value=0.01, value=float(example["two_days_ago"]), step=0.01
            )

        submitted = st.form_submit_button("Generate prediction", use_container_width=True)

with right_col:
    st.subheader("How it works")
    st.markdown(
        """
        - Choose an example or enter your own data
        - Select a model
        - Click predict
        - Review the result and price trend
        """
    )
    st.caption("Features used: " + str(metrics.get("feature_count", "n/a")))

if submitted:
    if not ticker:
        st.warning("Please enter a stock ticker.")
        st.stop()
    if not headline:
        st.warning("Please enter a headline.")
        st.stop()

    base_features, inferred_sentiment = build_single_input(
        ticker=ticker,
        headline=headline,
        today_price=float(today_price),
        one_day_ago=float(one_day_ago),
        two_days_ago=float(two_days_ago),
    )
    x_input = align_feature_columns(base_features, feature_columns)

    model = rf_model if chosen_model == "Random Forest" else lr_model
    pred = int(model.predict(x_input)[0])
    proba = float(model.predict_proba(x_input)[0][1]) if hasattr(model, "predict_proba") else None

    st.subheader("Prediction result")
    if pred == 1:
        st.error("Likely next-day price decrease")
    else:
        st.success("Likely stable or up next day")

    score_cols = st.columns(4)
    with score_cols[0]:
        st.metric("Model", chosen_model)
    with score_cols[1]:
        st.metric("Sentiment", inferred_sentiment.title())
    with score_cols[2]:
        st.metric("Drop probability", f"{proba:.2%}" if proba is not None else "n/a")
    with score_cols[3]:
        st.metric("Features", str(metrics.get("feature_count", "n/a")))

    chart_col, details_col = st.columns([1.1, 0.9], gap="large")
    price_trend = pd.DataFrame(
        {
            "Day": ["2 days ago", "1 day ago", "Today"],
            "Price": [float(two_days_ago), float(one_day_ago), float(today_price)],
        }
    )

    with chart_col:
        st.subheader("Recent price trend")
        st.line_chart(price_trend.set_index("Day"), height=260)

    with details_col:
        st.subheader("Details")
        st.write(
            {
                "Ticker": ticker,
                "Today price": round(float(today_price), 2),
                "1 day ago": round(float(one_day_ago), 2),
                "2 days ago": round(float(two_days_ago), 2),
            }
        )

    with st.expander("Engineered features"):
        st.dataframe(x_input.T.rename(columns={x_input.index[0]: "value"}), use_container_width=True)

else:
    st.info("Choose an example or enter your own values, then click `Generate prediction`.")
