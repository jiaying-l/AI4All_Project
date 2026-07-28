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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #07111f 0%, #0d1b2a 45%, #f4f7fb 45%, #f4f7fb 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(29,78,216,0.88));
            color: white;
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 22px 50px rgba(15, 23, 42, 0.25);
            margin-bottom: 1.25rem;
        }
        .glass-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 20px;
            padding: 1.2rem 1.2rem 0.8rem 1.2rem;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: white;
            border-radius: 18px;
            padding: 1rem;
            border-left: 5px solid #2563eb;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }
        .result-up {
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            border: 1px solid #86efac;
            border-radius: 22px;
            padding: 1.2rem;
            color: #14532d;
        }
        .result-down {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border: 1px solid #fca5a5;
            border-radius: 22px;
            padding: 1.2rem;
            color: #7f1d1d;
        }
        .small-label {
            font-size: 0.9rem;
            color: #475569;
            margin-bottom: 0.15rem;
        }
        .big-number {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">{label}</div>
            <div class="big-number">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

if not MODELS_DIR.exists():
    st.error("Model files are missing. Run: `python train_models.py`")
    st.stop()

rf_model, lr_model, feature_columns, metrics = load_artifacts()

hero_left, hero_right = st.columns([1.9, 1.1], gap="large")
with hero_left:
    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:0.95rem; letter-spacing:0.08em; text-transform:uppercase; opacity:0.8;">
                AI4ALL Final Presentation Demo
            </div>
            <h1 style="margin:0.35rem 0 0.5rem 0;">Next-Day Stock Drop Predictor</h1>
            <p style="font-size:1.08rem; max-width:44rem; line-height:1.6; margin-bottom:0;">
                Turn a financial headline and recent price movement into a simple, live risk signal.
                This interface is designed to make your machine learning workflow easy to explain on stage.
            </p>
        </div>
        """
        ,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown("### Demo controls")
    example_name = st.selectbox("Load a presentation example", list(EXAMPLES.keys()))
    chosen_model = st.radio("Prediction model", ["Random Forest", "Logistic Regression"], horizontal=True)
    st.markdown("Prediction label: `1 = likely next-day decrease`, `0 = likely stable or up`.")

example = EXAMPLES[example_name]

metric_cols = st.columns(3)
with metric_cols[0]:
    render_metric_card(
        "Random Forest accuracy",
        f"{metrics.get('random_forest_accuracy', 0.0):.2%}" if metrics else "n/a",
    )
with metric_cols[1]:
    render_metric_card(
        "Logistic Regression accuracy",
        f"{metrics.get('logistic_regression_accuracy', 0.0):.2%}" if metrics else "n/a",
    )
with metric_cols[2]:
    render_metric_card(
        "Features used",
        str(metrics.get("feature_count", "n/a")) if metrics else "n/a",
    )

left_col, right_col = st.columns([1.2, 0.8], gap="large")
with left_col:
    st.markdown("### Input market story")
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
    st.markdown("### What the app does")
    st.markdown(
        """
        - Reads the tone of the headline
        - Compares recent price momentum
        - Uses trained ML models to estimate downside risk
        - Shows the decision in a presentation-friendly format
        """
    )
    st.markdown("### Best demo tip")
    st.info(
        "Start with the built-in examples, then change one headline or one price input live to show how the prediction reacts."
    )

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

    st.markdown("## Prediction result")
    if pred == 1:
        st.markdown(
            """
            <div class="result-down">
                <div style="font-size:0.9rem; font-weight:600; text-transform:uppercase;">Risk signal</div>
                <div style="font-size:2rem; font-weight:800; margin:0.3rem 0;">Likely next-day price decrease</div>
                <div style="font-size:1rem;">The selected model sees downside pressure based on tone and recent momentum.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-up">
                <div style="font-size:0.9rem; font-weight:600; text-transform:uppercase;">Risk signal</div>
                <div style="font-size:2rem; font-weight:800; margin:0.3rem 0;">Likely stable or up next day</div>
                <div style="font-size:1rem;">The selected model does not see strong evidence of a next-day drop.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    result_col, details_col = st.columns([1.15, 0.85], gap="large")
    with result_col:
        score_cols = st.columns(3)
        with score_cols[0]:
            st.metric("Model used", chosen_model)
        with score_cols[1]:
            st.metric("Headline sentiment", inferred_sentiment.title())
        with score_cols[2]:
            st.metric("Drop probability", f"{proba:.2%}" if proba is not None else "n/a")

    with details_col:
        st.markdown("### Prediction breakdown")
        st.write(
            {
                "Ticker": ticker,
                "Today price": round(float(today_price), 2),
                "1 day ago": round(float(one_day_ago), 2),
                "2 days ago": round(float(two_days_ago), 2),
            }
        )
        price_trend = pd.DataFrame(
            {
                "Day": ["2 days ago", "1 day ago", "Today"],
                "Price": [float(two_days_ago), float(one_day_ago), float(today_price)],
            }
        )
        st.line_chart(price_trend.set_index("Day"))

    with st.expander("Engineered features used for this prediction"):
        st.dataframe(x_input.T.rename(columns={x_input.index[0]: "value"}), use_container_width=True)

else:
    st.markdown("## Ready for your demo")
    st.info("Choose an example or enter your own headline, then click `Generate prediction`.")
