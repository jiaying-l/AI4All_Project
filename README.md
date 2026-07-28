# AI4ALL Streamlit App

This project predicts whether a stock is likely to decrease the next day based on:
- a financial headline
- today's price
- prices from 1 and 2 days ago

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Files
- `app.py` - Streamlit user interface
- `train_models.py` - data download, training, and model export
- `ml_utils.py` - shared preprocessing and feature engineering
- `DEPLOYMENT.md` - deployment decisions and testing notes
