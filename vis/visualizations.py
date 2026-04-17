import logging
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

def run_visualizations():
    """
    Generate model comparison visualizations and save evaluation summary
    Reads:
        data/model_outputs/logistic_results.json
        data/model_outputs/random_forest_results.json
    Saves:
        vis/visualizations/accuracy_comparison.png
        data/model_evaluation/auc_comparison.png
        data/model_evaluation/model_evaluation_summary.csv"""

    os.makedirs("data/model_evaluation", exist_ok=True)
    #Load model results
    try:
        with open("data/model_outputs/logistic_results.json", "r") as f:
            lr = json.load(f)
        with open("data/model_outputs/random_forest_results.json", "r") as f:
            rf = json.load(f)
        logger.info("Model results loaded")
    except Exception as e:
        logger.error("Failed to load model results: %s", e)
        raise
    targets = ["Target_Daily", "Target_Monthly", "Target_Yearly"]
    labels = ["Daily", "Monthly", "Yearly"]
    #Accuracy comparison bar chart
    try:
        lr_acc = [lr["horizons"][t]["accuracy"] for t in targets]
        rf_acc = [rf["horizons"][t]["accuracy"] for t in targets]
        x = range(len(labels))
        plt.figure(figsize=(8, 5))
        plt.bar([i - 0.2 for i in x], lr_acc, 0.4, label="Logistic Regression")
        plt.bar([i + 0.2 for i in x], rf_acc, 0.4, label="Random Forest")
        plt.axhline(y=0.5, color="gray", linestyle="--", label="Baseline (50%)")
        plt.xlabel("Prediction Horizon")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy by Time Horizon")
        plt.xticks(x, labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig("vis/visualizations/accuracy_comparison.png", dpi=150)
        plt.close()
        logger.info("Accuracy comparison chart saved")
    except Exception as e:
        logger.error("Accuracy comparison chart failed: %s", e)
        raise
    #AUC comparison bar chart if roc_auc is present in results
    try:
        lr_auc = [lr["horizons"][t].get("roc_auc") for t in targets]
        rf_auc = [rf["horizons"][t].get("roc_auc") for t in targets]
        if all(v is not None for v in lr_auc + rf_auc):
            plt.figure(figsize=(8, 5))
            plt.bar([i - 0.2 for i in x], lr_auc, 0.4, label="Logistic Regression")
            plt.bar([i + 0.2 for i in x], rf_auc, 0.4, label="Random Forest")
            plt.axhline(y=0.5, color="gray", linestyle="--", label="Baseline (0.5)")
            plt.xlabel("Prediction Horizon")
            plt.ylabel("ROC-AUC")
            plt.title("Model ROC-AUC by Time Horizon")
            plt.xticks(x, labels)
            plt.legend()
            plt.tight_layout()
            plt.savefig("data/model_evaluation/auc_comparison.png", dpi=150)
            plt.close()
            logger.info("AUC comparison chart saved")
    except Exception as e:
        logger.error("AUC comparison chart failed: %s", e)
        raise
    #Summary CSV
    try:
        rows = []
        for t, label in zip(targets, labels):
            for model_name, results in [("Logistic Regression", lr), ("Random Forest", rf)]:
                h = results["horizons"][t]
                report = h.get("classification_report", {})
                rows.append({"Model": model_name, "Horizon": label, "Accuracy": round(h.get("accuracy", 0), 4), "ROC_AUC": round(h.get("roc_auc", 0), 4),
                    "Precision_Class1": round(report.get("1", {}).get("precision", 0), 4), "Recall_Class1": round(report.get("1", {}).get("recall", 0), 4), "F1_Class1": round(report.get("1", {}).get("f1-score", 0), 4),})
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv("data/model_evaluation/model_evaluation_summary.csv", index=False)
        logger.info("Evaluation summary CSV saved")
        print("Visualizations saved")
    except Exception as e:
        logger.error("Summary CSV generation failed: %s", e)
        raise

if __name__ == "__main__":
    run_visualizations()