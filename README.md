# Smart Expense Categorization + Spending Prediction

Upload a transactions CSV and get:

- **Categorization** — a TF-IDF + Logistic Regression classifier tags each
  transaction (Food, Transport, Shopping, Entertainment, Bills & Utilities,
  Groceries, Health, Travel, Education, Others).
- **Spending trend** — monthly totals as a bar chart.
- **Category distribution** — a pie chart of where the money goes.
- **Next-month prediction** — linear regression over monthly totals (or
  daily-average extrapolation if only one month of data is available).
- **Unusual spending** — transactions flagged by z-score within their category.
- **Budget warning** — compares the predicted next-month spend against a
  budget you set.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

No file needed to try it — the app falls back to `data/sample_transactions.csv`.

## CSV format

```
date,merchant,amount
2026-07-02,Swiggy,420
2026-07-03,Uber,230
```

- `date` is optional (defaults to today for every row, which disables the
  trend/prediction charts' usefulness — include it for real results).
- Column names are flexible: `description`/`narration`/`details` → merchant,
  `amt`/`value` → amount, `txn_date`/`transaction_date` → date.

## Files

- `app.py` — Streamlit UI
- `categorizer.py` — ML classifier + training data
- `predictor.py` — trend regression, anomaly detection, budget check
- `data/sample_transactions.csv` — demo dataset
