# register model

import json
import os
import warnings

import mlflow
import dagshub
from mlflow.exceptions import MlflowException

from src.logger import logging

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_DAGSHUB_TOKENTEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "pradeepbishnoi9601"
repo_name = "URLGuard"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# MLflow / DagsHub tracking setup (matches project convention)
# mlflow.set_tracking_uri("https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow")
# dagshub.init(repo_owner="pradeepbishnoi9601", repo_name="URLGuard", mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info (run_id + model_path) from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and alias it as 'staging'."""
    try:
        # Use the resolved model_uri saved at logging time (MLflow 3.x
        # Logged Model URI), not a reconstructed runs:/ path — reconstructing
        # it fails to resolve against the new Logged Model registry.
        model_uri = model_info["model_uri"]

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="staging",
                version=model_version.version,
            )
            logging.info(
                "Model %s version %s registered and aliased as 'staging'.",
                model_name,
                model_version.version,
            )
        except AttributeError:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
            )
            logging.info(
                "Model %s version %s registered and transitioned to Staging.",
                model_name,
                model_version.version,
            )

        return model_version

    except MlflowException as e:
        logging.error("MLflow error during model registration: %s", e)
        raise
    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise


def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "URLGuard_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()