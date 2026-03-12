import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

#Subject to change
#Performed worse at monthly, could be overfitting?
def run_random_forest():
    """
    Train and evaluate random forest across three time periods
    Reads:
        data/transformed/aapl_features.csv
    Saves:
        data/model_outputs/random_forest_results.json
    Returns:
        dict: Results for daily, monthly, and yearly targets"""

    #Load data
    df = pd.read_csv("data/transformed/aapl_features.csv", index_col="Date", parse_dates=True)

    #Define features
    feature_cols = ["Daily_Return", "MA_5", "MA_20", "Volatility_20", "Price_vs_MA5", "Price_vs_MA20", "Volume",
                    "CPI", "Unemployment_Rate", "Fed_Funds_Rate", "Treasury_10Y", "CPI_Change", "Unemployment_Change"]

    targets = ["Target_Daily", "Target_Monthly", "Target_Yearly"]
    all_results = {}

    for target in targets:
        print(f"\nRandom Forest: {target}")
        X = df[feature_cols]
        y = df[target]

        #Train/test split
        split_point = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        #Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        #Predictions
        y_pred = model.predict(X_test)

        #Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        #Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("Feature Importance:")
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        all_results[target] = {"accuracy": accuracy, "classification_report": report, "confusion_matrix": cm, "feature_importance": importance}

    #Save results
    results = {"model": "Random Forest", "feature_cols": feature_cols, "horizons": all_results}
    with open("data/model_outputs/random_forest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved")
    return results

if __name__ == "__main__":
    run_random_forest()