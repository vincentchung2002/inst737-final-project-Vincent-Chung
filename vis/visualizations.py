import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

#Currently have bar chart comparing the accuracy of the different models and time horizons
#Subject to add more
def run_visualizations():
    """
    Generate visualizations based on model results
    Reads:
        data/model_outputs/logistic_results.json
        data/model_outputs/random_forest_results.json
    Saves:
        vis/visualizations/"""

    #Load model results
    with open("data/model_outputs/logistic_results.json", "r") as f:
        lr = json.load(f)
    with open("data/model_outputs/random_forest_results.json", "r") as f:
        rf = json.load(f)
    targets = ["Target_Daily", "Target_Monthly", "Target_Yearly"]
    labels = ["Daily", "Monthly", "Yearly"]

    #Accuracy comparisons
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
    print("Visualizations saved")

if __name__ == "__main__":
    run_visualizations()