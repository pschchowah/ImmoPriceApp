import time
import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostModel:
    def __init__(self):
        """
        Initializes the XGBoostModel with an uninitialized model and dataset placeholder.
        """
        self.model = None
        self.X = None

    def train(self, X_train: pd.DataFrame,
              y_train: pd.Series) -> None:
        """
        Trains the XGBoost regression model on the provided training data.

        Args:
            X_train (pd.DataFrame): The input features for training.
            y_train (pd.Series): The target variable (output) for training.
        """
        start_time = time.time()
        self.X = X_train

        # Convert the training data to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Initialize and train the model
        self.model = xgb.train(
            params={'objective': 'reg:squarederror'},
            dtrain=dtrain)

        end_time = time.time()
        training_time = end_time - start_time
        print(
            f"Model training completed in {training_time:.4f} seconds.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained XGBoost model.

        Args:
            X (pd.DataFrame): The input features for making predictions.

        Returns:
            np.ndarray: The predicted values.
        """

        # Make predictions
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        predictions = self.model.predict(dmatrix)

        return predictions

    def save_model(self, model_path: str) -> None:
        """
        Saves the trained model to a specified file path.

        Args:
            model_path (str): The file path to save the model.
        """
        if self.model is not None:
            self.model.save_model(model_path)
            print(f"Model saved to {model_path}.")
        else:
            print("No model to save!")

    def load_model(self, model_path: str) -> None:
        """
        Loads a model from a specified file path.

        Args:
            model_path (str): The file path to load the model from.
        """
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}.")

