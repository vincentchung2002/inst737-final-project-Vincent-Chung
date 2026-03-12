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

## Code Package Structure

```
inst737-final-project-Vincent-Chung/
├── data/
│   ├── extracted/              
│   ├── transformed/           
│   ├── model_outputs/          
│   ├── eda/          
│   └── reference-tables/       
├── etl/
│   ├── extract.py             
│   ├── transform.py            
│   └── load.py                 
├── analysis/
│   ├── logistic_model.py       
│   └── random_forest_model.py  
├── vis/
│   └── visualizations.py       
├── main.py                     
├── README.md
└── requirements.txt
```