import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

from src.data_prep import load_and_preprocess

def predict(input_features: dict, model_type: str = "xgboost"):
    template = pd.read_csv("src/input_template.csv")

    input_df = pd.DataFrame([input_features])
    input_encoded = pd.get_dummies(input_df)

    input_row = template.copy()
    for col in input_encoded.columns:
        if col in input_row.columns:
            input_row.at[0, col] = input_encoded.at[0, col]

    # Load the selected model
    model_path = f"models/{model_type}.pkl"
    model = joblib.load(model_path)

    return round(model.predict(input_row)[0], 2)
