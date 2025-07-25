import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.comman_functions import read_yaml, load_data
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import joblib
import mlflow
import mlflow.sklearn

logger  = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_file_path):

        self.train_path = train_path
        self.test_path = test_path
        self.model_file_path = model_file_path
        
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info("Loading training and test data")

            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)

            logger.info("Data loaded successfully")

            X_train = train_data.drop(columns=["booking_status"])
            y_train = train_data["booking_status"]

            X_test = test_data.drop(columns=["booking_status"])
            y_test = test_data["booking_status"]

            logger.info("Data split into features and target variable")

            return X_train,y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error while loading and splitting data: {e}")
            raise CustomException("Failed to load and split data", e)
        
    def train_model(self, X_train, y_train):
        try:
            logger.info("Starting model training")

            model = lgb.LGBMClassifier()

            logger.info("Performing hyperparameter tuning using RandomizedSearchCV")

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )

            logger.info("Fitting the model with training data")

            random_search.fit(X_train, y_train)

            best_params = random_search.best_params_

            logger.info(f"Best parameters found: {best_params}")

            best_model = random_search.best_estimator_

            logger.info("Model training completed successfully")

            return best_model
        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to train model", e)
        
    def evaluate_model(self, model, X_test, y_test):
        try:

            logger.info("Starting model evaluation")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred )
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Model evaluation completed with accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Failed to evaluate model", e)
        
    def save_model(self, model):
        try:
            logger.info("Saving the trained model")

            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)

            logger.info(f"Model will be saved to {self.model_file_path}")

            joblib.dump(model, self.model_file_path)

            logger.info("Model saved successfully")

        except Exception as e:
            logger.error(f"Error while saving the model: {e}")
            raise CustomException("Failed to save model", e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process")
                logger.info("MLflow run started")

                mlflow.log_artifact(self.train_path, artifact_path="train_data")
                mlflow.log_artifact(self.test_path, artifact_path="test_data")

                X_train, y_train, X_test, y_test = self.load_and_split_data()

                model = self.train_model(X_train, y_train)

                metrics = self.evaluate_model(model, X_test, y_test)

                self.save_model(model)

                logger.info("Logging model and metrics to MLflow")
                mlflow.sklearn.log_model(model, "model")

                mlflow.log_artifact(self.model_file_path, artifact_path="model")
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model and metrics logged to MLflow")


                logger.info(f"Model training process completed successfully with metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error during model training process: {e}")
            raise CustomException("Failed during model training process", e)
        
if __name__ == "__main__":

    model = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_FILE_PATH)
    model.run()

