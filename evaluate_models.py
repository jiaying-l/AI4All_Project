"""
Model evaluation script — run after train_models.py.
Adds 5-fold cross-validation, classification reports, and confusion matrix plots.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ml_utils import load_and_prepare_data, prepare_features

LABELS = ["Did Not Decrease", "Decreased"]


def load_data():
    news_root = kagglehub.dataset_download(
        "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    )
    stock_root = kagglehub.dataset_download(
        "ehallmar/daily-historical-stock-prices-1970-2018"
    )
    news_csv = str(Path(news_root) / "analyst_ratings_processed.csv")
    stock_csv = str(Path(stock_root) / "historical_stock_prices.csv")

    bundle = load_and_prepare_data(news_csv, stock_csv)
    news = bundle.news.sample(n=min(50000, len(bundle.news)), random_state=6).reset_index(drop=True)
    return news


def main() -> None:
    models_dir = Path("models")

    rf_model = joblib.load(models_dir / "random_forest.joblib")
    lr_model = joblib.load(models_dir / "logistic_regression.joblib")
    feature_columns = joblib.load(models_dir / "feature_columns.joblib")

    print("Loading data...")
    news = load_data()

    X = news.drop(columns=["price_went_up"])
    y = news["price_went_up"]

    X_f = prepare_features(X)
    for col in feature_columns:
        if col not in X_f.columns:
            X_f[col] = 0.0
    X_f = X_f[feature_columns]

    # ── 5-fold stratified cross-validation ──────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n── Cross-Validation ────────────────────────────────────────────")
    for name, model in [("Random Forest", rf_model), ("Logistic Regression", lr_model)]:
        scores = cross_val_score(model, X_f, y, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"{name} (5-fold): {scores.round(4)}")
        print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")

    # ── Classification reports ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_f, y, test_size=0.2, random_state=6, stratify=y
    )
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)

    print("\n── Random Forest Classification Report ─────────────────────────")
    print(classification_report(y_test, y_pred_rf, target_names=LABELS))

    print("── Logistic Regression Classification Report ───────────────────")
    print(classification_report(y_test, y_pred_lr, target_names=LABELS))

    # ── Confusion matrix plots ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_pred, title, cmap in zip(
        axes,
        [y_pred_rf, y_pred_lr],
        ["Random Forest", "Logistic Regression"],
        ["Blues", "Oranges"],
    ):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=LABELS, yticklabels=LABELS, ax=ax,
        )
        ax.set_title(f"{title} — Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    out_path = models_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
