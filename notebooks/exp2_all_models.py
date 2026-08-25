import time
import logging
import warnings

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "PhiUSIIL_Phishing_URL_Dataset.csv",
    "test_size": 0.20,
    "random_state": 42,
    "mlflow_tracking_uri": "https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow",
    "dagshub_repo_owner": "pradeepbishnoi9601",
    "dagshub_repo_name": "URLGuard",
    "experiment_name": "Model Comparison - LR vs RF vs XGBoost vs LightGBM",
}

# ========================== FEATURE GROUPS ==========================
DROP_COLS = ["FILENAME", "URL", "Domain", "Title", "TLD", "Unnamed: 0"]
TARGET = "label"

NUMERIC = [
    "URLLength", "DomainLength", "URLSimilarityIndex", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain",
    "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL",
    "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "LineOfCode",
    "LargestLineLength", "DomainTitleMatchScore", "URLTitleMatchScore",
    "NoOfURLRedirect", "NoOfSelfRedirect", "NoOfPopup", "NoOfiFrame",
    "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef",
    "NoOfExternalRef",
]

BINARY_FLAGS = [
    "IsDomainIP", "HasObfuscation", "IsHTTPS", "HasTitle", "HasFavicon",
    "Robots", "IsResponsive", "HasDescription", "HasExternalFormSubmit",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField",
    "Bank", "Pay", "Crypto", "HasCopyrightInfo",
]

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(
    repo_owner=CONFIG["dagshub_repo_owner"],
    repo_name=CONFIG["dagshub_repo_name"],
    mlflow=True,
)
mlflow.set_experiment(CONFIG["experiment_name"])


# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Raw shape: {df.shape}")

        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            df = df.drop_duplicates()
            logging.info(f"Dropped {n_dupes} duplicate rows. New shape: {df.shape}")

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        unused = set(X.columns) - set(NUMERIC) - set(BINARY_FLAGS)
        if unused:
            logging.warning(f"Columns not assigned to a feature group: {unused}")

        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", BINARY_FLAGS),
        ],
        remainder="drop",
    )


# ========================== MODELS ==========================
ALGORITHMS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=CONFIG["random_state"],
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=CONFIG["random_state"],
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=CONFIG["random_state"],
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=CONFIG["random_state"],
        verbosity=-1,
    ),
}


# XGBoost/LightGBM wrap non-sklearn-native C++ Booster objects. MLflow's
# sklearn flavor now checks every object type before pickling a Pipeline,
# so these third-party types must be explicitly allow-listed.
SKOPS_TRUSTED_TYPES = {
    "XGBoost": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
    "LightGBM": [
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
    ],
}


def log_model_params(algo_name, classifier):
    """Logs the key hyperparameters of the trained classifier to MLflow."""
    params_to_log = {}

    if algo_name == "LogisticRegression":
        params_to_log["max_iter"] = classifier.max_iter
        params_to_log["class_weight"] = str(classifier.class_weight)

    elif algo_name == "RandomForest":
        params_to_log["n_estimators"] = classifier.n_estimators
        params_to_log["max_depth"] = classifier.max_depth
        params_to_log["class_weight"] = str(classifier.class_weight)

    elif algo_name == "XGBoost":
        params_to_log["n_estimators"] = classifier.n_estimators
        params_to_log["learning_rate"] = classifier.learning_rate
        params_to_log["max_depth"] = classifier.max_depth

    elif algo_name == "LightGBM":
        params_to_log["n_estimators"] = classifier.n_estimators
        params_to_log["learning_rate"] = classifier.learning_rate
        params_to_log["max_depth"] = classifier.max_depth
        params_to_log["class_weight"] = str(classifier.class_weight)

    mlflow.log_params(params_to_log)


# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_state"],
    )

    logging.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    with mlflow.start_run(run_name="All Experiments") as parent_run:

        mlflow.log_params({
            "test_size": CONFIG["test_size"],
            "n_train_rows": X_train.shape[0],
            "n_test_rows": X_test.shape[0],
            "n_features": X.shape[1],
        })

        for algo_name, classifier in ALGORITHMS.items():
            with mlflow.start_run(run_name=algo_name, nested=True) as child_run:
                try:
                    start_time = time.time()

                    logging.info(f"Training {algo_name}...")

                    model = Pipeline(steps=[
                        ("preprocessor", build_preprocessor()),
                        ("classifier", classifier),
                    ])

                    mlflow.log_param("algorithm", algo_name)

                    model.fit(X_train, y_train)

                    log_model_params(algo_name, classifier)

                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, zero_division=0),
                        "roc_auc": roc_auc_score(y_test, y_proba),
                        "pr_auc": average_precision_score(y_test, y_proba),
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                        "true_positives": int(tp),
                        "training_time_seconds": time.time() - start_time,
                    }
                    mlflow.log_metrics(metrics)

                    trusted_types = SKOPS_TRUSTED_TYPES.get(algo_name)
                    if trusted_types:
                        mlflow.sklearn.log_model(
                            model, "model", skops_trusted_types=trusted_types
                        )
                    else:
                        mlflow.sklearn.log_model(model, "model")

                    print(f"\nAlgorithm: {algo_name}")
                    print(f"Metrics: {metrics}")

                except Exception as e:
                    print(f"Error in training {algo_name}: {e}")
                    mlflow.log_param("error", str(e))


# ========================== EXECUTION ==========================
if __name__ == "__main__":
    X, y = load_data(CONFIG["data_path"])
    train_and_evaluate(X, y)