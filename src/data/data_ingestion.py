# data ingestion

import os

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import yaml
from sklearn.model_selection import train_test_split

from src.logger import logging
# from src.connections import s3_connection  # wire in once the S3 source is ready


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load the PhiUSIIL dataset from a local CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logging.error('Data file not found: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light, pre-split cleaning only — deduping and label validation.
    Column dropping / scaling / encoding happens later in
    data_preprocessing.py and feature_engineering.py, fit on the
    training split only, to avoid leaking test-set statistics.
    """
    try:
        logging.info("pre-processing...")

        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            df = df.drop_duplicates()
            logging.info('Dropped %d duplicate rows', n_dupes)

        if 'label' not in df.columns:
            raise KeyError("'label' column not found in dataframe")

        n_before = len(df)
        final_df = df[df['label'].isin([0, 1])].copy()
        n_dropped = n_before - len(final_df)
        if n_dropped > 0:
            logging.warning('Dropped %d rows with an invalid/missing label', n_dropped)

        final_df['label'] = final_df['label'].astype(int)

        logging.info('Data preprocessing completed. Shape: %s', final_df.shape)
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test splits under <data_path>/raw/."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        # Local file for now — swap to the commented S3 block below once
        # the bucket/credentials are set up.
        data_url = params['data_ingestion']['data_path']
        df = load_data(data_url=data_url)

        # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("PhiUSIIL_Phishing_URL_Dataset.csv")

        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            stratify=final_df['label'],
            random_state=42,
        )

        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()