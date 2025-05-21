import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from data_prep import load_and_preprocess
import os
import joblib

def train_and_log(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Save both locally and to MLflow
        save_path = f"models/{model_name}.pkl"
        joblib.dump(model, save_path)
        mlflow.sklearn.log_model(model, model_name)

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("housing-price-comparison")

    os.makedirs("models", exist_ok=True)

    # Train and log both models
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    train_and_log(xgb_model, "xgboost", X_train, y_train, X_test, y_test)

    lin_model = LinearRegression()
    train_and_log(lin_model, "linear", X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    train_model()
