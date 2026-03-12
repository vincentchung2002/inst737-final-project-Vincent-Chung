import pandas as pd
import yfinance as yf
from fredapi import Fred

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
    stock_df = yf.download("AAPL", start=START, end=END, progress=False)
    stock_df.reset_index(inplace=True)
    stock_df.to_csv("data/extracted/aapl_daily.csv", index=False)
    print("Stock extraction complete")

    #FRED data
    fred = Fred(api_key=FRED_API_KEY)
    fred_series = {
        "CPIAUCSL": "CPI", #Consumer Price Index
        "UNRATE": "Unemployment_Rate", #Unemployment Rate
        "FEDFUNDS": "Fed_Funds_Rate", #Federal Funds Rate
        "DGS10": "Treasury_10Y", #10-Year Treasury Yield
        }
    econ_data = {}
    for series_id, col_name in fred_series.items():
        econ_data[col_name] = fred.get_series(series_id, observation_start=START, observation_end=END)
    econ_df = pd.DataFrame(econ_data)
    econ_df.index.name = "Date"
    econ_df.reset_index(inplace=True)
    econ_df.to_csv("data/extracted/fred_economic_data.csv", index=False)
    print("FRED extraction complete")
    return stock_df, econ_df

if __name__ == "__main__":
    extract_data()