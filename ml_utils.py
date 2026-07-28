from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SENTIMENT_MAP = {"positive": 1, "neutral": 0, "negative": -1}

# Small lexicon-based sentiment fallback for stable deployment.
POSITIVE_WORDS = {
    "beats",
    "beat",
    "growth",
    "gain",
    "gains",
    "upgrade",
    "upgrades",
    "strong",
    "surge",
    "record",
    "profit",
    "profits",
    "bullish",
    "outperform",
}
NEGATIVE_WORDS = {
    "miss",
    "misses",
    "drop",
    "drops",
    "fall",
    "falls",
    "downgrade",
    "downgrades",
    "weak",
    "loss",
    "losses",
    "lawsuit",
    "bearish",
    "underperform",
}


@dataclass
class DataBundle:
    news: pd.DataFrame
    stock: pd.DataFrame


def simple_headline_sentiment(headline: str) -> str:
    tokens = {t.strip(".,!?;:()[]{}\"'").lower() for t in headline.split()}
    pos_count = len(tokens.intersection(POSITIVE_WORDS))
    neg_count = len(tokens.intersection(NEGATIVE_WORDS))
    if pos_count > neg_count:
        return "positive"
    if neg_count > pos_count:
        return "negative"
    return "neutral"


def load_and_prepare_data(news_csv: str, stock_csv: str) -> DataBundle:
    news = pd.read_csv(news_csv)
    stock = pd.read_csv(stock_csv)

    if "Unnamed: 0" in news.columns:
        news = news.drop(columns=["Unnamed: 0"])

    news = news.dropna().drop_duplicates()
    news = news[news["title"].str.contains(" ", na=False)]
    news["date"] = news["date"].astype(str).str.split(" ").str[0]

    stock = stock.dropna().drop_duplicates()
    stock["date"] = stock["date"].astype(str)

    news_start, news_end = news["date"].min(), news["date"].max()
    stock = stock[(stock["date"] >= news_start) & (stock["date"] <= news_end)]

    stock_start, stock_end = stock["date"].min(), stock["date"].max()
    news = news[(news["date"] >= stock_start) & (news["date"] <= stock_end)]

    common_tickers = set(news["stock"].unique()).intersection(stock["ticker"].unique())
    news = news[news["stock"].isin(common_tickers)].copy()
    stock = stock[stock["ticker"].isin(common_tickers)].copy()

    stock["date"] = pd.to_datetime(stock["date"])
    news["date"] = pd.to_datetime(news["date"])
    stock["price"] = stock["close"]

    lookup = stock.set_index(["ticker", "date"])["price"].to_dict()
    news["today_price"] = [lookup.get((t, d)) for t, d in zip(news["stock"], news["date"])]
    news["1ago_price"] = [
        lookup.get((t, d - timedelta(days=1))) for t, d in zip(news["stock"], news["date"])
    ]
    news["2ago_price"] = [
        lookup.get((t, d - timedelta(days=2))) for t, d in zip(news["stock"], news["date"])
    ]
    news["nextday_price"] = [
        lookup.get((t, d + timedelta(days=1))) for t, d in zip(news["stock"], news["date"])
    ]
    news = news.dropna().copy()

    for col in ["today_price", "1ago_price", "2ago_price", "nextday_price"]:
        news[col] = news[col].round(4)

    news["price_went_up"] = (news["today_price"] > news["nextday_price"]).astype(int)
    news["sentitment"] = news["title"].apply(simple_headline_sentiment)

    return DataBundle(news=news, stock=stock)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    sent_col = "sentitment"
    if {"stock", "date", "title"}.issubset(f.columns):
        daily_n = f.groupby(["stock", "date"])["title"].transform("count")
        avg_n = f.groupby("stock")["title"].transform("count") / max(f["date"].nunique(), 1)
        sent = f[sent_col].map(SENTIMENT_MAP).fillna(0) if sent_col in f.columns else 0
        f["news_surge_ratio"] = daily_n / avg_n.replace(0, 1)
        f["competitive_pressure_index"] = f["news_surge_ratio"] * (sent * -1)

    f = f.drop(columns=[c for c in ["nextday_price", "title", "date", "stock"] if c in f.columns])
    if sent_col in f.columns:
        f["sentiment_score"] = f[sent_col].map(SENTIMENT_MAP).fillna(0)
        f = f.drop(columns=[sent_col])

    if {"today_price", "1ago_price"}.issubset(f.columns):
        f["price_change_1d"] = f["today_price"] - f["1ago_price"]
        f["price_pct_change_1d"] = (f["price_change_1d"] / f["1ago_price"].replace(0, np.nan)).fillna(
            0
        )
    if {"today_price", "2ago_price"}.issubset(f.columns):
        f["price_change_2d"] = f["today_price"] - f["2ago_price"]

    return f.fillna(0)


def build_single_input(
    ticker: str,
    headline: str,
    today_price: float,
    one_day_ago: float,
    two_days_ago: float,
    sentiment: str | None = None,
) -> Tuple[pd.DataFrame, str]:
    if sentiment is None:
        sentiment = simple_headline_sentiment(headline)

    row = pd.DataFrame(
        [
            {
                "stock": ticker,
                "date": pd.Timestamp("today").normalize(),
                "title": headline,
                "today_price": today_price,
                "1ago_price": one_day_ago,
                "2ago_price": two_days_ago,
                "nextday_price": today_price,
                "sentitment": sentiment,
            }
        ]
    )

    features = prepare_features(row)
    return features, sentiment


def align_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[feature_columns]
