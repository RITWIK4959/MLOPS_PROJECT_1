from src.data_ingestion import DataIngestion
from src.data_preprocess import DataProcessor
from src.model_training import ModelTraining
from utils.comman_functions import read_yaml
from config.paths_config import *

if __name__ == "__main__":

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()

    model = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_FILE_PATH)
    model.run()

