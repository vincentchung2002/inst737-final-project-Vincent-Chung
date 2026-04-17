import logging
from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.logistic_model import run_logistic
from analysis.random_forest_model import run_random_forest
from vis.visualizations import run_visualizations

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting extraction")
        extract_data()
        logger.info("Extraction complete")
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        raise
    try:
        logger.info("Starting transformation")
        transform_data()
        logger.info("Transformation complete")
    except Exception as e:
        logger.error("Transformation failed: %s", e)
        raise
    try:
        logger.info("Starting load")
        load_data()
        logger.info("Load complete")
    except Exception as e:
        logger.error("Load failed: %s", e)
        raise
    try:
        logger.info("Starting logistic regression")
        run_logistic()
        logger.info("Logistic regression complete")
    except Exception as e:
        logger.error("Logistic regression failed: %s", e)
        raise
    try:
        logger.info("Starting random forest")
        run_random_forest()
        logger.info("Random forest complete")
    except Exception as e:
        logger.error("Random forest failed: %s", e)
        raise
    try:
        logger.info("Starting visualizations")
        run_visualizations()
        logger.info("Visualizations complete")
    except Exception as e:
        logger.error("Visualizations failed: %s", e)
        raise

if __name__ == "__main__":
    main()