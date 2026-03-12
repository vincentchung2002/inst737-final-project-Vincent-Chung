from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.logistic_model import logistic
from analysis.random_forest_model import random_forest
from vis.visualizations import visualizations

def main():
    extract_data()
    transform_data()
    load_data()
    logistic()
    random_forest()
    visualizations()

if __name__ == "__main__":
    main()