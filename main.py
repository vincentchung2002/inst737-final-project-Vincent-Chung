from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.logistic_model import run_logistic
from analysis.random_forest_model import run_random_forest
from vis.visualizations import run_visualizations

def main():
    extract_data()
    transform_data()
    load_data()
    run_logistic()
    run_random_forest()
    run_visualizations()

if __name__ == "__main__":
    main()