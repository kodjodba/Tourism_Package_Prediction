# ------------------------------------------------------------------
# Model training with experiment tracking (production / CI version)
# ------------------------------------------------------------------
# for data manipulation
import os
import json
import numpy as np
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Tourism_Package_Prediction")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation",
]

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

# Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [200, 300],
    "xgbclassifier__max_depth": [5, 7],
    "xgbclassifier__colsample_bytree": [0.7, 0.9],
    "xgbclassifier__colsample_bylevel": [0.7, 0.9],
    "xgbclassifier__learning_rate": [0.1, 0.2],
    "xgbclassifier__reg_lambda": [1, 3],
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="xgb_gridsearch_pipeline"):
    # Hyperparameter tuning with GridSearchCV (optimise F1 for the buyer class)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run,
    # so all experiments can be compared side by side in the MLflow UI
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold: maximise out-of-fold F1 on the training data
    oof_proba = cross_val_predict(best_model, Xtrain, ytrain, cv=cv, method="predict_proba")[:, 1]
    candidates = np.round(np.arange(0.30, 0.76, 0.05), 2)
    classification_threshold = float(max(candidates, key=lambda t: f1_score(ytrain, (oof_proba >= t).astype(int))))
    mlflow.log_param("classification_threshold", classification_threshold)

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
        "test_roc_auc": roc_auc_score(ytest, y_pred_test_proba),
    }
    mlflow.log_metrics(metrics)

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # Save the serving configuration (threshold + metrics) used by the Streamlit app
    config_path = "tourism_project/deployment/model_config.json"
    with open(config_path, "w") as f:
        json.dump({"classification_threshold": classification_threshold,
                   "best_params": grid_search.best_params_,
                   "test_metrics": {k: round(float(v), 4) for k, v in metrics.items() if k.startswith("test")}},
                  f, indent=2)
    mlflow.log_artifact(config_path, artifact_path="model")

    print("Best parameters:", grid_search.best_params_)
    print(f"Classification threshold: {classification_threshold}")
    print(classification_report(ytest, y_pred_test))
    print({k: round(float(v), 3) for k, v in metrics.items()})
    print(f"Model saved to {model_path}")
