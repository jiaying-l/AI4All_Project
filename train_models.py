from __future__ import annotations

import json
from pathlib import Path

import joblib
import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml_utils import load_and_prepare_data, prepare_features


def dataset_paths() -> tuple[str, str]:
    news_root = kagglehub.dataset_download(
        "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    )
    stock_root = kagglehub.dataset_download("ehallmar/daily-historical-stock-prices-1970-2018")
    return (
        str(Path(news_root) / "analyst_ratings_processed.csv"),
        str(Path(stock_root) / "historical_stock_prices.csv"),
    )


def main() -> None:
    news_csv, stock_csv = dataset_paths()
    bundle = load_and_prepare_data(news_csv, stock_csv)
    news = bundle.news.sample(n=min(50000, len(bundle.news)), random_state=6).reset_index(drop=True)

    X = news.drop(columns=["price_went_up"])
    y = news["price_went_up"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=6, stratify=y
    )

    X_train_f = prepare_features(X_train)
    X_test_f = prepare_features(X_test)
    feature_columns = X_train_f.columns.tolist()

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_model.fit(X_train_f, y_train)
    rf_pred = rf_model.predict(X_test_f)

    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(X_train_f, y_train)
    lr_pred = lr_model.predict(X_test_f)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_model, models_dir / "random_forest.joblib")
    joblib.dump(lr_model, models_dir / "logistic_regression.joblib")
    joblib.dump(feature_columns, models_dir / "feature_columns.joblib")

    metrics = {
        "random_forest_accuracy": float(accuracy_score(y_test, rf_pred)),
        "logistic_regression_accuracy": float(accuracy_score(y_test, lr_pred)),
        "train_rows": int(len(X_train_f)),
        "test_rows": int(len(X_test_f)),
        "feature_count": int(len(feature_columns)),
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete. Saved files:")
    print("- models/random_forest.joblib")
    print("- models/logistic_regression.joblib")
    print("- models/feature_columns.joblib")
    print("- models/metrics.json")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
