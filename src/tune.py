import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/abdosakr127/MLOPS.mlflow/")

# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/abdosakr127/MLOPS.mlflow/")

# Load train data
train = pd.read_csv("data/train.csv")
X = train.drop("species", axis=1)
y = train["species"]

# Hyperparameters to test
n_estimators_list = [10, 50, 100]
max_depth_list = [None, 5, 10]

# Parent run for hyperparameter tuning
with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):
    for n_est in n_estimators_list:
        for max_d in max_depth_list:
            # Nested run for each combination
            with mlflow.start_run(run_name=f"n_est_{n_est}_max_d_{max_d}", nested=True):
                model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
                scores = cross_val_score(model, X, y, cv=3)
                mean_score = scores.mean()
                std_score = scores.std()

                # Log parameters
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", max_d)

                # Log metrics
                mlflow.log_metric("mean_cv_score", mean_score)
                mlflow.log_metric("std_cv_score", std_score)

                # Log the model
                mlflow.sklearn.log_model(model, "model")