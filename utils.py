import os

from google.cloud import storage
from settings import logger, headers

import pandas as pd

def download_file(bucket_name, name, filename):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(name)
    try:
        logger.debug('Downloading data from storage')
        blob.download_to_filename(filename)
        logger.info(f"Downloaded file from {bucket_name}")
        return True
    except Exception as e:
        logger.error("Exception ", e)


def list_blobs(bucket_name, pattern=None):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    if pattern:
        files = [blob.name for blob in blobs if pattern in blob.name]

    else:
        files = [blob.name for blob in blobs]

    return files


def load_txt(filename, names=None) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=';', encoding='latin1', header=None)
    df = df.drop(columns=[df.shape[1] - 1])
    if names:
        df.columns = names
    else:
        basename = os.path.basename(filename)
        basename = '_'.join(basename.split('_')[:2])
        df.columns = headers[basename]

    return df
