import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

#Subject to change
#Strongest model at the monthly time horizon (54%)
def run_logistic():
    """
    Train and evaluate logistic regression across three time horizons
    Reads:
        data/transformed/aapl_features.csv
    Saves:
        data/model_outputs/logistic_results.json
    Returns:
        dict: Results for daily, monthly, and yearly targets"""

    #Load data
    df = pd.read_csv("data/transformed/aapl_features.csv", index_col="Date", parse_dates=True)

    #Define features
    feature_cols = ["Daily_Return", "MA_5", "MA_20", "Volatility_20", "Price_vs_MA5", "Price_vs_MA20", "Volume",
                    "CPI", "Unemployment_Rate", "Fed_Funds_Rate", "Treasury_10Y","CPI_Change", "Unemployment_Change"]

    targets = ["Target_Daily", "Target_Monthly", "Target_Yearly"]
    all_results = {}
    for target in targets:
        print(f"\nLogistic Regression: {target}")
        X = df[feature_cols]
        y = df[target]

        #Train/test split
        split_point = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        #Pre-processing (scale features)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        #Predictions
        y_pred = model.predict(X_test_scaled)

        #Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        all_results[target] = {"accuracy": accuracy, "classification_report": report, "confusion_matrix": cm}

    #Save results
    results = {"model": "Logistic Regression", "feature_cols": feature_cols, "horizons": all_results}
    with open("data/model_outputs/logistic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved")
    return results

if __name__ == "__main__":
    run_logistic()