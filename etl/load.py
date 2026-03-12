import pandas as pd
import sqlite3

def load_data():
    """
    Validate transformed data and load into SQLite
    Reads:
        data/transformed/aapl_features.csv
    Saves:
        data/transformed/project.db (SQLite database)
    Returns:
        pd.DataFrame: Final dataset after validation"""

    #Read transformed data
    df = pd.read_csv("data/transformed/aapl_features.csv", index_col="Date", parse_dates=True)
    #Validation
    null_count = df.isnull().sum().sum()
    dup_count = df.index.duplicated().sum()
    if null_count > 0:
        print(f"{null_count} null values found")
    if dup_count > 0:
        print(f"{dup_count} duplicate dates found")
        
    #Save to SQLite
    conn = sqlite3.connect("data/transformed/project.db")
    df.to_sql("aapl_features", conn, if_exists="replace", index=True)
    conn.close()
    print("Data loaded to SQLite")
    return df

if __name__ == "__main__":
    load_data()