import logging
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
logger = logging.getLogger(__name__)

#Performed worse at monthly, could be overfitting
def run_random_forest():
    """
    Train and evaluate random forest across three time periods
    Reads:
        data/transformed/aapl_features.csv
    Saves:
        data/model_outputs/random_forest_results.json
        data/model_evaluation/ (confusion matrix heatmaps, ROC curves, feature importance per horizon)
    Returns:
        dict: Results for daily, monthly, and yearly targets"""
        
    os.makedirs("data/model_evaluation", exist_ok=True)
    #Load data
    try:
        df = pd.read_csv("data/transformed/aapl_features.csv", index_col="Date", parse_dates=True)
        logger.info("Random Forest: data loaded: %d rows", len(df))
    except Exception as e:
        logger.error("Random Forest: failed to load data: %s", e)
        raise
    #Define features
    feature_cols = ["Daily_Return", "MA_5", "MA_20", "Volatility_20", "Price_vs_MA5", "Price_vs_MA20", "Volume",
                    "CPI", "Unemployment_Rate", "Fed_Funds_Rate", "Treasury_10Y", "CPI_Change", "Unemployment_Change"]
    targets = ["Target_Daily", "Target_Monthly", "Target_Yearly"]
    labels = ["Daily", "Monthly", "Yearly"]
    all_results = {}

    for target, label in zip(targets, labels):
        try:
            logger.info("Random Forest: training for %s", target)
            X = df[feature_cols]
            y = df[target]
            #Train/test split
            split_point = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            #Train model (no scaling needed — RF is scale-invariant)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            #Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            #Evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred).tolist()
            #Feature importance
            importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
            logger.info("Random Forest %s — Accuracy: %.4f, ROC-AUC: %.4f", target, accuracy, roc_auc)
            print(f"\nRandom Forest: {target}")
            print(f"Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}")
            print(classification_report(y_test, y_pred))
            print("Feature Importance:")
            for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {imp:.4f}")
            #Confusion matrix heatmap
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
            plt.title(f"Random Forest — Confusion Matrix ({label})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(f"data/model_evaluation/rf_confusion_matrix_{label.lower()}.png", dpi=150)
            plt.close()
            #ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], "k--", label="Baseline")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve — Random Forest ({label})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"data/model_evaluation/rf_roc_curve_{label.lower()}.png", dpi=150)
            plt.close()
            #Feature importance bar chart
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            feat_names, feat_vals = zip(*sorted_importance)
            plt.figure(figsize=(8, 5))
            plt.barh(feat_names[::-1], feat_vals[::-1])
            plt.xlabel("Importance")
            plt.title(f"Random Forest — Feature Importance ({label})")
            plt.tight_layout()
            plt.savefig(f"data/model_evaluation/rf_feature_importance_{label.lower()}.png", dpi=150)
            plt.close()
            all_results[target] = {"accuracy": accuracy, "roc_auc": roc_auc, "classification_report": report, "confusion_matrix": cm, "feature_importance": importance}
            logger.info("Random Forest: evaluation plots saved for %s", target)
        except Exception as e:
            logger.error("Random Forest: failed for %s: %s", target, e)
            raise
    #Save results
    try:
        results = {"model": "Random Forest", "feature_cols": feature_cols, "horizons": all_results}
        with open("data/model_outputs/random_forest_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Random Forest: results saved")
        print("Results saved")
    except Exception as e:
        logger.error("Random Forest: failed to save results: %s", e)
        raise
    return results

if __name__ == "__main__":
    run_random_forest()