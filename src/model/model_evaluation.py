# model evaluation

import os
import json
import pickle

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.logger import logging

# for the local setup 

# -------------------------------------------------------------------------------------
# MLflow / DagsHub tracking setup (matches project convention)
# mlflow.set_tracking_uri("https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow")
# dagshub.init(repo_owner="pradeepbishnoi9601", repo_name="URLGuard", mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "pradeepbishnoi9601"
repo_name = "URLGuard"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the data: %s", e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
        logging.info("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise


def get_feature_importance(clf, feature_names: list, file_path: str) -> pd.DataFrame:
    """
    Compute and save XGBoost feature importances, sorted descending.
    This directly targets the unresolved 'perfect score' question — if
    URLSimilarityIndex / TLDLegitimateProb dominate, that's the smoking gun.
    """
    try:
        importances = clf.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        fi_df.to_csv(file_path, index=False)
        logging.info("Feature importances saved to %s", file_path)
        return fi_df
    except Exception as e:
        logging.error("Error computing feature importances: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved to %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the metrics: %s", e)
        raise


def save_model_info(run_id: str, model_uri: str, file_path: str) -> None:
    """Save the model run ID and resolved model URI to a JSON file."""
    try:
        model_info = {"run_id": run_id, "model_uri": model_uri}
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logging.debug("Model info saved to %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the model info: %s", e)
        raise


def main():
    mlflow.set_experiment("XGBoost - Final Model")
    with mlflow.start_run() as run:
        try:
            clf = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_features.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            feature_names = test_data.columns[:-1].tolist()

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, "reports/metrics.json")

            fi_df = get_feature_importance(
                clf, feature_names, "reports/feature_importance.csv"
            )

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model params to MLflow
            if hasattr(clf, "get_params"):
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Log model — skops_trusted_types required for XGBoost Booster
            logged_model = mlflow.sklearn.log_model(
                clf,
                name="model",  # MLflow 3.x: use name=, not the deprecated artifact_path positional arg
                skops_trusted_types=[
                    "xgboost.sklearn.XGBClassifier",
                    "xgboost.core.Booster",
                ],
            )

            # Save + log model info
            save_model_info(run.info.run_id, logged_model.model_uri, "reports/experiment_info.json")

            # Log artifacts
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("reports/feature_importance.csv")

            # Flag suspiciously perfect scores directly in the run
            if metrics["accuracy"] >= 0.9999:
                logging.warning(
                    "Accuracy is ~1.0 — check reports/feature_importance.csv "
                    "for a possible leaky proxy feature (e.g. URLSimilarityIndex, "
                    "TLDLegitimateProb) before treating this run as final."
                )
                mlflow.set_tag("suspicious_perfect_score", "true")

        except Exception as e:
            logging.error("Failed to complete the model evaluation process: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()