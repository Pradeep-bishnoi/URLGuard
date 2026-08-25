# load test + signature test + performance test

import unittest
import os
import pandas as pd
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Same tracking setup as model_evaluation.py / register_model.py /
        # scripts/promote_model.py — already confirmed working in CI.
        # mlflow.set_tracking_uri("https://dagshub.com/pradeepbishnoi9601/URLGuard.mlflow")
        # dagshub.init(repo_owner="pradeepbishnoi9601", repo_name="URLGuard", mlflow=True)

        # for the production use
         # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "pradeepbishnoi9601"
        repo_name = "URLGuard"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        cls.model_name = "URLGuard_model"

        # Load via alias, not the deprecated stages=["Staging"] API —
        # register_model.py registers new versions under "staging".
        cls.model_uri = f"models:/{cls.model_name}@staging"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # No vectorizer in this project — feature_engineering.py already
        # produced fully processed/scaled features, so the holdout file
        # can be fed to the model directly with no extra transform step.
        cls.holdout_data = pd.read_csv("data/processed/test_features.csv")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        # Structural check rather than a vectorizer-based feature-name
        # check (no vectorizer here) — confirms the model accepts the
        # processed feature matrix as-is and returns one prediction per
        # input row.
        X_sample = self.holdout_data.iloc[:5, :-1]
        predictions = self.model.predict(X_sample)

        self.assertEqual(len(predictions), X_sample.shape[0])
        # Single prediction per row (binary classification) — not one
        # column per class.
        self.assertEqual(len(getattr(predictions, "shape", (len(predictions),))), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.model.predict(X_holdout)

        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred)
        recall = recall_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred)

        # NOTE — these thresholds are a placeholder, not a validated
        # target. The project's own handoff flags accuracy/F1 sitting at
        # ~1.0 as a likely leakage artifact (URLSimilarityIndex /
        # TLDLegitimateProb acting as a near-direct proxy for the
        # label) — see reports/feature_importance.csv. Until that's
        # root-caused, a high threshold here mostly confirms the model
        # still has the same leak, not that it's actually good.
        # Revisit this threshold once the leakage question is resolved.
        expected_accuracy = 0.90
        expected_precision = 0.90
        expected_recall = 0.90
        expected_f1 = 0.90

        self.assertGreaterEqual(accuracy, expected_accuracy, f"Accuracy should be at least {expected_accuracy}")
        self.assertGreaterEqual(precision, expected_precision, f"Precision should be at least {expected_precision}")
        self.assertGreaterEqual(recall, expected_recall, f"Recall should be at least {expected_recall}")
        self.assertGreaterEqual(f1, expected_f1, f"F1 score should be at least {expected_f1}")


if __name__ == "__main__":
    unittest.main()