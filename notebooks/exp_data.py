"""
URLGuard - Phishing URL Detection
Baseline Model: Logistic Regression
--------------------------------------------------
Dataset: PhiUSIIL Phishing URL Dataset
Target : label  (1 = legitimate, 0 = phishing — verify against dataset docs)
"""

import time
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

import dagshub
import mlflow
import mlflow.sklearn


# --------------------------------------------------
# Configure Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------------------------------
# 0. Load & Prepare Data
# --------------------------------------------------

logging.info("Loading dataset...")

df = pd.read_csv(r'C:\Users\prade\Desktop\URLGuard\notebooks\exp_data.csv')

logging.info(f"Raw shape: {df.shape}")

# --------------------------------------------------
# 0a. Data validation checks (log, don't silently trust the data)
# --------------------------------------------------

n_dupes = df.duplicated().sum()
n_missing = df.isnull().sum().sum()
class_balance = df["label"].value_counts(normalize=True).to_dict()

logging.info(f"Duplicate rows: {n_dupes}")
logging.info(f"Missing values (total cells): {n_missing}")
logging.info(f"Class balance: {class_balance}")

if n_dupes > 0:
    df = df.drop_duplicates()
    logging.info(f"Dropped duplicates. New shape: {df.shape}")

# --------------------------------------------------
# 0b. Drop non-feature columns
# --------------------------------------------------
# FILENAME/URL/Domain/Title are raw identifiers/text, not usable directly.
# TLD is high-cardinality categorical; TLDLegitimateProb + TLDLength already
# capture its signal numerically, so we drop the raw column to avoid a
# leakage-prone high-cardinality encoding in the baseline.

drop_cols = ["FILENAME", "URL", "Domain", "Title", "TLD"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

target = "label"
X = df.drop(columns=[target])
y = df[target]

# --------------------------------------------------
# 0c. Feature groups
# --------------------------------------------------

numeric = [
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

binary_flags = [
    "IsDomainIP", "HasObfuscation", "IsHTTPS", "HasTitle", "HasFavicon",
    "Robots", "IsResponsive", "HasDescription", "HasExternalFormSubmit",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField",
    "Bank", "Pay", "Crypto", "HasCopyrightInfo",
]

# sanity check: every column in X should be accounted for
unused = set(X.columns) - set(numeric) - set(binary_flags)
if unused:
    logging.warning(f"Columns present in X but not assigned to a group: {unused}")

# --------------------------------------------------
# 0d. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

logging.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")


# --------------------------------------------------
# DagsHub / MLflow Setup
# --------------------------------------------------

mlflow.set_tracking_uri("https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow")
dagshub.init(repo_owner="pradeepbishnoi9601", repo_name="URLGuard", mlflow=True)

mlflow.set_experiment("Logistic Regression Baseline")


# --------------------------------------------------
# Start MLflow Run
# --------------------------------------------------

with mlflow.start_run():

    start_time = time.time()

    try:

        # --------------------------------------------------
        # 1. Log Preprocessing Parameters
        # --------------------------------------------------

        logging.info("Logging preprocessing parameters...")

        mlflow.log_param("numeric_features", ", ".join(numeric))
        mlflow.log_param("binary_flag_features", ", ".join(binary_flags))
        mlflow.log_param("dropped_columns", ", ".join(drop_cols))
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("n_features_total", X.shape[1])
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("n_train_rows", X_train.shape[0])
        mlflow.log_param("n_test_rows", X_test.shape[0])

        # --------------------------------------------------
        # 2. Create Preprocessor
        # --------------------------------------------------

        logging.info("Creating preprocessing pipeline...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("bin", "passthrough", binary_flags),
            ],
            remainder="drop"
        )

        # --------------------------------------------------
        # 3. Create Complete Pipeline
        # --------------------------------------------------

        logging.info("Creating Logistic Regression pipeline...")

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]
        )

        # --------------------------------------------------
        # 4. Log Model Parameters
        # --------------------------------------------------

        logging.info("Logging model parameters...")

        mlflow.log_param("model", "Logistic Regression")

        mlflow.log_params({
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        })

        # --------------------------------------------------
        # 5. Train Model
        # --------------------------------------------------

        logging.info("Fitting the model...")

        model.fit(X_train, y_train)

        logging.info("Model training complete.")

        # --------------------------------------------------
        # 6. Predictions
        # --------------------------------------------------

        logging.info("Making predictions...")

        y_pred = model.predict(X_test)

        # Probability of the positive class (verify which label index = phishing)
        y_proba = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # 7. Calculate Metrics
        # --------------------------------------------------

        logging.info("Calculating evaluation metrics...")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # --------------------------------------------------
        # 8. Log Metrics to MLflow
        # --------------------------------------------------

        logging.info("Logging evaluation metrics...")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        mlflow.log_metric("true_negatives", int(tn))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("true_positives", int(tp))

        # --------------------------------------------------
        # 9. Log Complete Pipeline
        # --------------------------------------------------

        logging.info("Saving and logging complete pipeline...")

        mlflow.sklearn.log_model(model, "model")

        # --------------------------------------------------
        # 10. Training Time
        # --------------------------------------------------

        end_time = time.time()
        training_time = end_time - start_time

        mlflow.log_metric("training_time_seconds", training_time)

        logging.info(
            f"Model training and logging completed "
            f"in {training_time:.2f} seconds."
        )

        # --------------------------------------------------
        # 11. Print Results
        # --------------------------------------------------

        logging.info(f"Accuracy : {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall   : {recall:.4f}")
        logging.info(f"F1 Score : {f1:.4f}")
        logging.info(f"ROC-AUC  : {roc_auc:.4f}")
        logging.info(f"PR-AUC   : {pr_auc:.4f}")

        print("\n========== MODEL RESULTS ==========")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"PR-AUC   : {pr_auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN={tn}  FP={fp}")
        print(f"  FN={fn}  TP={tp}")
        print("\n", classification_report(y_test, y_pred, digits=4))

        print("MLflow Run ID:")
        print(mlflow.active_run().info.run_id)
        print("===================================")

    except Exception as e:

        logging.error(
            f"An error occurred: {e}",
            exc_info=True
        )

        raise