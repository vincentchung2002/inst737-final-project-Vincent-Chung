import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def transform_data():
    """
    Clean, merge, engineer features, and EDA.
    Reads:
        data/extracted/aapl_daily.csv
        data/extracted/fred_economic_data.csv
    Saves:
        data/transformed/aapl_features.csv
        data/visualizations/eda/
    Returns:
        pd.DataFrame: Final analytical dataset ready for modeling"""

    #Raw data
    stock_df = pd.read_csv("data/extracted/aapl_daily.csv", header=[0, 1])
    stock_df.columns = stock_df.columns.get_level_values(0)
    econ_df = pd.read_csv("data/extracted/fred_economic_data.csv")

    #Clean stock data
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df.set_index("Date", inplace=True)
    stock_df.sort_index(inplace=True)
    if "Adj Close" in stock_df.columns:
        stock_df.drop(columns=["Adj Close"], inplace=True)
    for col in stock_df.columns:
        stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")
    stock_df = stock_df[~stock_df.index.duplicated(keep="first")]
    stock_df.dropna(inplace=True)
    print(f"Stock data cleaned")

    #Clean economic data
    econ_df["Date"] = pd.to_datetime(econ_df["Date"])
    econ_df.set_index("Date", inplace=True)
    econ_df.sort_index(inplace=True)
    econ_df = econ_df[~econ_df.index.duplicated(keep="first")]
    print(f"Economic data cleaned")

    #Align and merge
    daily_dates = pd.date_range(start=econ_df.index.min(), end=econ_df.index.max(), freq="D")
    econ_daily = econ_df.reindex(daily_dates).ffill()
    econ_daily.index.name = "Date"
    merged_df = stock_df.join(econ_daily, how="inner")
    print(f"Dataset merged")

    #Feature engineering 
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

    #Target variable
    merged_df["Target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)
    #Drop NaN rows and shift
    merged_df.dropna(inplace=True)

    #EDA - Closing price over time
    plt.figure(figsize=(12, 5))
    plt.plot(merged_df.index, merged_df["Close"], linewidth=0.8)
    plt.title("AAPL Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig("data/visualizations/eda/aapl_closing_price.png", dpi=150)
    plt.close()

    #EDA - Closing price with moving averages
    plt.figure(figsize=(12, 5))
    plt.plot(merged_df.index, merged_df["Close"], label="Close", linewidth=0.8)
    plt.plot(merged_df.index, merged_df["MA_5"], label="5-Day MA", linewidth=0.8)
    plt.plot(merged_df.index, merged_df["MA_20"], label="20-Day MA", linewidth=0.8)
    plt.title("AAPL Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/visualizations/eda/aapl_moving_averages.png", dpi=150)
    plt.close()

    #EDA - Correlation matrix
    feature_cols = ["Close", "Volume", "Daily_Return", "MA_5", "MA_20", "Volatility_20", "Price_vs_MA5", "Price_vs_MA20",
                    "CPI", "Unemployment_Rate", "Fed_Funds_Rate", "Treasury_10Y", "CPI_Change", "Unemployment_Change", "Target"]
    plt.figure(figsize=(12, 10))
    sns.heatmap(merged_df[feature_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("data/visualizations/eda/correlation_matrix.png", dpi=150)
    plt.close()

    #EDA - Daily returns distribution
    plt.figure(figsize=(10, 5))
    plt.hist(merged_df["Daily_Return"], bins=100, edgecolor="black", linewidth=0.3)
    plt.title("Distribution of AAPL Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("data/visualizations/eda/daily_returns_distribution.png", dpi=150)
    plt.close()

    #EDA - Target variable balance
    plt.figure(figsize=(6, 4))
    merged_df["Target"].value_counts().plot(kind="bar", color=["salmon", "steelblue"])
    plt.title("Target Variable Distribution")
    plt.xlabel("0 = Down, 1 = Up")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("data/visualizations/eda/target_distribution.png", dpi=150)
    plt.close()
    print("EDA plots saved")

    #Final dataset
    merged_df.to_csv("data/transformed/aapl_features.csv")
    print("Transformed dataset saved")
    return merged_df

if __name__ == "__main__":
    transform_data()