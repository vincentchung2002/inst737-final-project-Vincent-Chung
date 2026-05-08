import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
logger = logging.getLogger(__name__)

def extract_data():
    """
    Extract all raw data needed
    Pulls:
        AAPL daily stock data (Open, High, Low, Close, Volume) from Yahoo Finance
        CPI, Unemployment Rate, Fed Funds Rate, 10-Year Treasury Yield from FRED
    Saves:
        data/extracted/aapl_daily.csv
        data/extracted/fred_economic_data.csv
    Returns:
        tuple: (stock_df, econ_df) as pd DataFrames"""

    FRED_API_KEY = "b0bc43f50ab348b518f4de426a6cd09c"
    #Date range
    START = "2015-01-01"
    END = "2026-03-12"
    #AAPL data
    try:
        stock_df = yf.download("AAPL", start=START, end=END, progress=False)
        stock_df.reset_index(inplace=True)
        stock_df.to_csv("data/extracted/aapl_daily.csv", index=False)
        logger.info("Stock extraction complete: %d rows", len(stock_df))
    except Exception as e:
        logger.error("Stock extraction failed: %s", e)
        raise
    #FRED data
    try:
        fred = Fred(api_key=FRED_API_KEY)
        fred_series = {"CPIAUCSL": "CPI", "UNRATE": "Unemployment_Rate", "FEDFUNDS": "Fed_Funds_Rate", "DGS10": "Treasury_10Y", }
        econ_data = {}
        for series_id, col_name in fred_series.items():
            econ_data[col_name] = fred.get_series(series_id, observation_start=START, observation_end=END)
        econ_df = pd.DataFrame(econ_data)
        econ_df.index.name = "Date"
        econ_df.reset_index(inplace=True)
        econ_df.to_csv("data/extracted/fred_economic_data.csv", index=False)
        logger.info("FRED extraction complete: %d rows", len(econ_df))
    except Exception as e:
        logger.error("FRED extraction failed: %s", e)
        raise
    return stock_df, econ_df

if __name__ == "__main__":
    extract_data()