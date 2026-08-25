# data preprocessing

import os

import numpy as np
import pandas as pd

from src.logger import logging

# FILENAME/URL/Domain/Title are raw identifiers/text, not usable directly.
# TLD is high-cardinality categorical; TLDLegitimateProb + TLDLength already
# capture its signal numerically, so the raw column is dropped to avoid a
# leakage-prone high-cardinality encoding.
# "Unnamed: 0" is a stray pandas index column from an earlier CSV export
# that didn't use index=False.
DROP_COLS = ["FILENAME", "URL", "Domain", "Title", "TLD", "Unnamed: 0"]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-feature identifier/text columns from a PhiUSIIL dataframe.

    Unlike a text pipeline, there's no free-text column here to clean
    (stopwords/lemmatization/etc.) — the URL was already engineered into
    numeric/binary features upstream. This step's job is narrower:
    strip columns that aren't model features and aren't the target,
    leaving scaling/encoding for feature_engineering.py.

    Args:
        df (pd.DataFrame): Raw PhiUSIIL dataframe (one split — train or test).

    Returns:
        pd.DataFrame: Dataframe with identifier/text columns removed.
    """
    try:
        present = [c for c in DROP_COLS if c in df.columns]
        if present:
            df = df.drop(columns=present)
            logging.info('Dropped non-feature columns: %s', present)

        missing_target = 'label' not in df.columns
        if missing_target:
            raise KeyError("'label' column not found in dataframe")

        logging.info("Data pre-processing completed. Shape: %s", df.shape)
        return df
    except KeyError as e:
        logging.error('Missing expected column: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data)
        test_processed_data = preprocess_dataframe(test_data)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()