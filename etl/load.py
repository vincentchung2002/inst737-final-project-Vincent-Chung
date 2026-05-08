import logging
import pandas as pd
import sqlite3
logger = logging.getLogger(__name__)

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
    try:
        df = pd.read_csv("data/transformed/aapl_features.csv", index_col="Date", parse_dates=True)
        logger.info("Transformed data loaded: %d rows", len(df))
    except Exception as e:
        logger.error("Failed to load transformed data: %s", e)
        raise
    #Validation
    try:
        null_count = df.isnull().sum().sum()
        dup_count = df.index.duplicated().sum()
        if null_count > 0:
            logger.warning("%d null values found", null_count)
        if dup_count > 0:
            logger.warning("%d duplicate dates found", dup_count)
        logger.info("Validation complete — nulls: %d, duplicate dates: %d", null_count, dup_count)
    except Exception as e:
        logger.error("Validation failed: %s", e)
        raise
    #Save to SQLite
    try:
        conn = sqlite3.connect("data/transformed/project.db")
        df.to_sql("aapl_features", conn, if_exists="replace", index=True)
        conn.close()
        logger.info("Data loaded to SQLite")
    except Exception as e:
        logger.error("SQLite load failed: %s", e)
        raise
    return df

if __name__ == "__main__":
    load_data()