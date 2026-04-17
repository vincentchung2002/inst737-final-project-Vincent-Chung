import logging
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
logger = logging.getLogger(__name__)

#Strongest model at the monthly time horizon (54%)
def run_logistic():
    """
    Train and evaluate logistic regression across three time horizons
    Reads:
        data/transformed/aapl_features.csv
    Saves:
        data/model_outputs/logistic_results.json
        data/model_evaluation/ (confusion matrix heatmaps, ROC curves per horizon)
    Returns:
        dict: Results for daily, monthly, and yearly targets"""

    os.makedirs("data/model_evaluation", exist_ok=True)

    #Load data
    try:
        df = pd.read_csv("data/transformed/aapl_features.csv", index_col="Date", parse_dates=True)
        logger.info("Logistic: data loaded: %d rows", len(df))
    except Exception as e:
        logger.error("Logistic: failed to load data: %s", e)
        raise

    #Define features
    feature_cols = ["Daily_Return", "MA_5", "MA_20", "Volatility_20", "Price_vs_MA5", "Price_vs_MA20", "Volume",
                    "CPI", "Unemployment_Rate", "Fed_Funds_Rate", "Treasury_10Y", "CPI_Change", "Unemployment_Change"]
    targets = ["Target_Daily", "Target_Monthly", "Target_Yearly"]
    labels = ["Daily", "Monthly", "Yearly"]
    all_results = {}
    
    for target, label in zip(targets, labels):
        try:
            logger.info("Logistic: training for %s", target)
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
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            #Evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred).tolist()
            logger.info("Logistic %s — Accuracy: %.4f, ROC-AUC: %.4f", target, accuracy, roc_auc)
            print(f"\nLogistic Regression: {target}")
            print(f"Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}")
            print(classification_report(y_test, y_pred))

            #Confusion matrix heatmap
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
            plt.title(f"Logistic Regression — Confusion Matrix ({label})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(f"data/model_evaluation/lr_confusion_matrix_{label.lower()}.png", dpi=150)
            plt.close()

            #ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], "k--", label="Baseline")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve — Logistic Regression ({label})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"data/model_evaluation/lr_roc_curve_{label.lower()}.png", dpi=150)
            plt.close()

            all_results[target] = {"accuracy": accuracy, "roc_auc": roc_auc, "classification_report": report, "confusion_matrix": cm}
            logger.info("Logistic: evaluation plots saved for %s", target)
        except Exception as e:
            logger.error("Logistic: failed for %s: %s", target, e)
            raise

    #Save results
    try:
        results = {"model": "Logistic Regression", "feature_cols": feature_cols, "horizons": all_results}
        with open("data/model_outputs/logistic_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Logistic: results saved")
        print("Results saved")
    except Exception as e:
        logger.error("Logistic: failed to save results: %s", e)
        raise
    return results

if __name__ == "__main__":
    run_logistic()