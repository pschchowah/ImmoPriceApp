import xgboost as xgb
import pandas as pd


class XGBoostPredictor:
    def __init__(self, model_path):
        # Load the trained model from the specified path
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, new_data):
        # Ensure new_data is in the correct format (DataFrame)
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError(
                "new_data must be a pandas DataFrame")

        # Convert new_data to DMatrix format required by XGBoost
        dmatrix = xgb.DMatrix(new_data)

        # Make the prediction
        prediction = self.model.predict(dmatrix)
        return prediction


# Example of usage:
if __name__ == "__main__":
    # Path to your trained model
    model_path = 'your_model.json'  # Update with your model path

    # Create instance of the predictor
    predictor = XGBoostPredictor(model_path)

    # Prepare new data for prediction
    new_data = pd.DataFrame({
        'feature1': [value1],
        'feature2': [value2],
        # Add more features as required
    })

    # Get prediction
    prediction = predictor.predict(new_data)
    print("Predicted price: ", prediction)

