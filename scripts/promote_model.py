# promote model

import os
import mlflow
import dagshub

# -------------------------------------------------------------------------------------
# MLflow / DagsHub tracking setup — same convention as
# src/model/model_evaluation.py and src/model/register_model.py.
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri("https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow")
# dagshub.init(repo_owner="pradeepbishnoi9601", repo_name="URLGuard", mlflow=True)
# -------------------------------------------------------------------------------------


def promote_model():

    # used for the production 
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

    client = mlflow.MlflowClient()
    model_name = "URLGuard_model"

    # Get the version currently aliased "staging" — NOT the deprecated
    # stages=["Staging"] API. register_model.py registers new versions
    # under the "staging" alias, so that's what we read here.
    staging_version = client.get_model_version_by_alias(model_name, "staging")

    # Note the current "production" version, if any, purely for the log
    # line below — reassigning an alias with set_registered_model_alias
    # automatically moves it off whatever version held it before, no
    # separate "archive" step is needed under the alias-based API.
    try:
        previous_prod = client.get_model_version_by_alias(model_name, "production")
        previous_prod_version = previous_prod.version
    except mlflow.exceptions.MlflowException:
        previous_prod_version = None

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=staging_version.version,
    )

    if previous_prod_version:
        print(
            f"Model '{model_name}' version {staging_version.version} promoted to "
            f"'production' (previously version {previous_prod_version})."
        )
    else:
        print(
            f"Model '{model_name}' version {staging_version.version} promoted to "
            f"'production' (no prior production version)."
        )


if __name__ == "__main__":
    promote_model()