# feature engineering

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.logger import logging

# NUMERIC gets StandardScaler; BINARY_FLAGS (already 0/1) pass through
# unchanged. No equivalent of BoW's max_features here — the feature set
# is fixed by the PhiUSIIL schema, not a tunable hyperparameter, so this
# module doesn't read from params.yaml.
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

TARGET = "label"


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)

        n_missing = df.isnull().sum().sum()
        if n_missing > 0:
            logging.warning('%d missing values found in %s', n_missing, file_path)

        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_feature_engineering(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """
    Fit a StandardScaler(numeric) + passthrough(binary flags)
    ColumnTransformer on the training split, and apply the same fitted
    transform to both splits. XGBoost is the final model for this
    project — scaling isn't strictly required for tree-based models,
    but keeping the shared preprocessor consistent across every
    candidate model (including the earlier LR baseline) keeps the
    pipeline reusable if the model choice ever changes.
    """
    try:
        logging.info("Applying feature engineering...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC),
                ("bin", "passthrough", BINARY_FLAGS),
            ],
            remainder="drop",
        )

        X_train = train_data[NUMERIC + BINARY_FLAGS]
        y_train = train_data[TARGET].values
        X_test = test_data[NUMERIC + BINARY_FLAGS]
        y_test = test_data[TARGET].values

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        feature_names = NUMERIC + BINARY_FLAGS

        train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        test_df['label'] = y_test

        os.makedirs('models', exist_ok=True)
        pickle.dump(preprocessor, open('models/preprocessor.pkl', 'wb'))

        logging.info('Feature engineering applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logging.error('Error during feature engineering transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_feature_engineering(train_data, test_data)

        save_data(train_df, os.path.join("./data", "processed", "train_features.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_features.csv"))
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()