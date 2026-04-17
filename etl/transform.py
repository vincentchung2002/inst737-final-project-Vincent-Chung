import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
logger = logging.getLogger(__name__)

def transform_data():
    """
    Clean, merge, engineer features, and EDA
    Reads:
        data/extracted/aapl_daily.csv
        data/extracted/fred_economic_data.csv
    Saves:
        data/transformed/aapl_features.csv
        data/eda/
    Returns:
        pd.DataFrame: Final analytical dataset ready for modeling"""

    #Raw data
    try:
        stock_df = pd.read_csv("data/extracted/aapl_daily.csv", header=[0, 1])
        stock_df.columns = stock_df.columns.get_level_values(0)
        econ_df = pd.read_csv("data/extracted/fred_economic_data.csv")
        logger.info("Raw data loaded: %d stock rows, %d econ rows", len(stock_df), len(econ_df))
    except Exception as e:
        logger.error("Failed to load raw data: %s", e)
        raise
    #Clean stock data
    try:
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])
        stock_df.set_index("Date", inplace=True)
        stock_df.sort_index(inplace=True)
        if "Adj Close" in stock_df.columns:
            stock_df.drop(columns=["Adj Close"], inplace=True)
        for col in stock_df.columns:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")
        stock_df = stock_df[~stock_df.index.duplicated(keep="first")]
        stock_df.dropna(inplace=True)
        logger.info("Stock data cleaned: %d rows", len(stock_df))
    except Exception as e:
        logger.error("Stock data cleaning failed: %s", e)
        raise
    #Clean economic data
    try:
        econ_df["Date"] = pd.to_datetime(econ_df["Date"])
        econ_df.set_index("Date", inplace=True)
        econ_df.sort_index(inplace=True)
        econ_df = econ_df[~econ_df.index.duplicated(keep="first")]
        logger.info("Economic data cleaned: %d rows", len(econ_df))
    except Exception as e:
        logger.error("Economic data cleaning failed: %s", e)
        raise
    #Align and merge (forward-fill monthly FRED data to daily frequency)
    try:
        daily_dates = pd.date_range(start=econ_df.index.min(), end=econ_df.index.max(), freq="D")
        econ_daily = econ_df.reindex(daily_dates).ffill()
        econ_daily.index.name = "Date"
        merged_df = stock_df.join(econ_daily, how="inner")
        logger.info("Dataset merged: %d rows", len(merged_df))
    except Exception as e:
        logger.error("Merge failed: %s", e)
        raise
    #Feature engineering
    try:
        #Daily return
        merged_df["Daily_Return"] = merged_df["Close"].pct_change()
        #Moving averages
        merged_df["MA_5"] = merged_df["Close"].rolling(window=5).mean()
        merged_df["MA_20"] = merged_df["Close"].rolling(window=20).mean()
        #Rolling volatility (20-day)
        merged_df["Volatility_20"] = merged_df["Daily_Return"].rolling(window=20).std()
        #Price relative to moving averages
        merged_df["Price_vs_MA5"] = merged_df["Close"] / merged_df["MA_5"]
        merged_df["Price_vs_MA20"] = merged_df["Close"] / merged_df["MA_20"]
        #Monthly percentage changes in economic indicators
        merged_df["CPI_Change"] = merged_df["CPI"].pct_change(periods=20)
        merged_df["Unemployment_Change"] = merged_df["Unemployment_Rate"].pct_change(periods=20)
        #Target variables (daily, monthly (20 days), yearly (252 days))
        #Daily
        merged_df["Target_Daily"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)
        #Monthly
        merged_df["Target_Monthly"] = (merged_df["Close"].shift(-20) > merged_df["Close"]).astype(int)
        #Yearly
        merged_df["Target_Yearly"] = (merged_df["Close"].shift(-252) > merged_df["Close"]).astype(int)
        #Drop NaN rows and shifts
        merged_df.dropna(inplace=True)
        logger.info("Feature engineering complete: %d rows, %d columns", len(merged_df), len(merged_df.columns))
    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        raise
    #EDA plots
    try:
        #Closing price over time
        plt.figure(figsize=(12, 5))
        plt.plot(merged_df.index, merged_df["Close"], linewidth=0.8)
        plt.title("AAPL Closing Price")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.tight_layout()
        plt.savefig("data/eda/aapl_closing_price.png", dpi=150)
        plt.close()
        #Closing price with moving averages
        plt.figure(figsize=(12, 5))
        plt.plot(merged_df.index, merged_df["Close"], label="Close", linewidth=0.8)
        plt.plot(merged_df.index, merged_df["MA_5"], label="5-Day MA", linewidth=0.8)
        plt.plot(merged_df.index, merged_df["MA_20"], label="20-Day MA", linewidth=0.8)
        plt.title("AAPL Price with Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/eda/aapl_moving_averages.png", dpi=150)
        plt.close()
        #Correlation matrix
        feature_cols = ["Close", "Volume", "Daily_Return", "MA_5", "MA_20", "Volatility_20", "Price_vs_MA5", "Price_vs_MA20",
                        "CPI", "Unemployment_Rate", "Fed_Funds_Rate", "Treasury_10Y", "CPI_Change", "Unemployment_Change", "Target_Daily", "Target_Monthly", "Target_Yearly"]
        plt.figure(figsize=(12, 10))
        sns.heatmap(merged_df[feature_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig("data/eda/correlation_matrix.png", dpi=150)
        plt.close()
        #Daily returns distribution
        plt.figure(figsize=(10, 5))
        plt.hist(merged_df["Daily_Return"], bins=100, edgecolor="black", linewidth=0.3)
        plt.title("Distribution of AAPL Daily Returns")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("data/eda/daily_returns_distribution.png", dpi=150)
        plt.close()
        #Target variable balance
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, target in zip(axes, ["Target_Daily", "Target_Monthly", "Target_Yearly"]):
            merged_df[target].value_counts().plot(kind="bar", color=["salmon", "steelblue"], ax=ax)
            ax.set_title(target)
            ax.set_xlabel("0 = Down, 1 = Up")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig("data/eda/target_distribution.png", dpi=150)
        plt.close()
        logger.info("EDA plots saved")
    except Exception as e:
        logger.error("EDA plot generation failed: %s", e)
        raise
    #Final dataset
    try:
        merged_df.to_csv("data/transformed/aapl_features.csv")
        logger.info("Transformed dataset saved: %d rows, %d columns", len(merged_df), len(merged_df.columns))
    except Exception as e:
        logger.error("Failed to save transformed dataset: %s", e)
        raise
    return merged_df

if __name__ == "__main__":
    transform_data()