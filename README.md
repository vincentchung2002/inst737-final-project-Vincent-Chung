# AAPL Stock Price Direction Prediction

## Project Overview

**Business Problem:** Investors and traders often struggle with making consistent, data-driven decisions in the stock market. This project aims to predict the short-term (daily) and long-term (monthly/yearly) price direction (up or down) of Apple (AAPL) stock using historical price data and macroeconomic indicators. The client is a portfolio manager at an investment firm that provides investment advice to average, median income clients.

**Datasets:**
- **AAPL Daily Stock Data** — Open, High, Low, Close, Volume (source: Yahoo Finance via yfinance)
- **Consumer Price Index (CPI)** — (source: FRED CPIAUCSL)
- **Unemployment Rate** — (source: FRED UNRATE)
- **Federal Funds Rate** — (source: FRED FEDFUNDS)
- **10-Year Treasury Yield** — (source: FRED DGS10)

**Techniques:**
- Binary classification (1 = price goes up, 0 = price goes down)
- Feature engineering (daily returns, moving averages, rolling volatility, price-to-MA ratios, economic indicator changes)
- Logistic Regression (baseline model)
- Random Forest Classifier (nonlinear model)
- Multi-horizon comparison (daily, monthly, yearly)
- Model evaluation: accuracy, ROC-AUC, precision, recall, F1, confusion matrices, ROC curves
- Pipeline logging via Python `logging` module with output to `pipeline.log`
- Error handling via `try/except` blocks at each ETL and modeling stage

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/vincentchung2002/inst737-final-project-Vincent-Chung.git
cd inst737-final-project-Vincent-Chung
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

Run the full pipeline:
```bash
python main.py
```

Pipeline logs are written to `pipeline.log` in the project root.

## Code Package Structure

```
inst737-final-project-Vincent-Chung/
├── data/
│   ├── extracted/              # Raw data from yfinance and FRED
│   ├── transformed/            # Cleaned, merged, and feature-engineered data + SQLite DB
│   ├── model_outputs/          # Model results JSON (accuracy, ROC-AUC, classification reports)
│   ├── model_evaluation/       # Evaluation charts: confusion matrices, ROC curves, feature importance, summary CSV
│   ├── eda/                    # Exploratory data analysis plots
│   └── reference-tables/       # Data dictionaries for all datasets
├── etl/
│   ├── extract.py              # Pulls raw data from yfinance and FRED API
│   ├── transform.py            # Cleans, merges, engineers features, generates EDA plots
│   └── load.py                 # Validates data and loads into SQLite
├── analysis/
│   ├── logistic_model.py       # Logistic regression training and evaluation
│   └── random_forest_model.py  # Random forest training and evaluation
├── vis/
│   ├── visualizations/         # Model comparison charts
│   └── visualizations.py       # Generates accuracy/AUC comparison charts and summary CSV
├── main.py                     # Orchestrates full pipeline with logging and error handling
├── pipeline.log                # Runtime log output (generated on run)
├── README.md
└── requirements.txt
```