# Strategic Trend Analysis Pipeline

This project downloads Google Trends data for a **semantic core** of keywords (expanded from a base word like `love`), forecasts yearly seasonality patterns using **Facebook Prophet**, clusters similar patterns, and generates a **PDF report** with predicted peak dates for the current year.

The script is fully **local** (no Jupyter required) and caches downloaded data as CSV files to avoid repeated network requests.

---

## ✨ Features

- **Semantic expansion** – Uses DataMuse API + Google Trends suggestions to build a relevant keyword set.
- **Local CSV caching** – Each keyword’s raw Trends data is saved (e.g., `data/love.csv`). If the file already covers the required date range (from 2018 to last year), it is reused.
- **Prophet forecasting** – Trains a model on historical data **up to the previous year** (excludes current year) and extracts the normalized yearly component.
- **Peak detection** – Identifies dates with high seasonal influence using `scipy.signal.find_peaks`.
- **Clustering** – Groups keywords with similar yearly patterns using K‑Means (max 3 clusters).
- **PDF report** – Contains a title page, one plot per cluster, and a table of predicted peak dates.

---

## 📦 Requirements

- Python 3.8 or higher
- The following Python packages (see `requirements.txt`):
