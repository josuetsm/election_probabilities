import os

from google.cloud import storage
from settings import logger, headers

import pandas as pd

from datetime import datetime


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


def get_candidates_info():
    candidates = load_txt('data/docs/Escenario_Candidatos_018032.txt')
    candidates = candidates[candidates['cod_elec'] == 8]
    servel_imfd = pd.read_csv('data/docs/pactos_servel_imfd.csv')
    cand2pact = dict(zip(candidates['cod_cand'], candidates['cod_pacto']))
    pact2imfd = dict(zip(servel_imfd['COD_PACTO'], servel_imfd['COD_IMFD']))
    pact2imfd[175] = 10  # D26
    pact2imfd[176] = 10  # D28
    cod2glosa_imfd = dict(zip(servel_imfd['COD_IMFD'], servel_imfd['GLOSA_IMFD']))

    return candidates, cand2pact, pact2imfd, cod2glosa_imfd


def get_time():
    df = pd.read_csv('data/docs/datatransfer2_info.csv')

    df['atime'] = df['atime'].apply(lambda x: datetime.fromtimestamp(x))
    df['mtime'] = df['mtime'].apply(lambda x: datetime.fromtimestamp(x))

    df = df[df['name'].apply(lambda x: 'VOTACION_8_' in x)]
    df['atime_labels'] = df['atime'].apply(lambda x: x.strftime("%H:%M"))
    df = df[3:-1].reset_index(drop=True)

    return df

