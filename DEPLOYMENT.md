# Streamlit Deployment Notes

## What was deployed
- A Streamlit app in `app.py` that accepts a stock ticker, one headline, and recent prices.
- A training/export script in `train_models.py` that trains two models and saves artifacts in `models/`.
- Shared preprocessing helpers in `ml_utils.py` so training and inference use the same logic.

## Why these decisions
- **Two models kept**: Random Forest and Logistic Regression to match the notebook and allow comparison.
- **Saved model artifacts**: avoids retraining each app start, making the app fast and reliable.
- **Lightweight sentiment fallback**: uses lexicon-based headline sentiment for easier deployment and fewer heavy dependencies.

## Setup requirements
- Python 3.10+ (recommended)
- Internet access for first-time Kaggle dataset download

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train models
Run once to generate model files:

```bash
python train_models.py
```

Expected outputs:
- `models/random_forest.joblib`
- `models/logistic_regression.joblib`
- `models/feature_columns.joblib`
- `models/metrics.json`

## Run the app locally

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Test plan used
- Entered realistic headline + price values for a large-cap ticker.
- Verified:
  - app loads without missing file errors after training
  - both model selections return predictions
  - probability output appears for supported model
  - input validation blocks empty ticker/headline

## Known limitations
- Current sentiment step uses a lexicon fallback, not full FinBERT inference.
- Prediction quality depends on the downloaded dataset and may vary by market regime.
