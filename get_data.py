import os
from utils import download_file, list_blobs

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/constitucion-josue.json'

bucket_name = 'election_day'
bucket_files = list_blobs(bucket_name, pattern='VOTACION_8_')

len(bucket_files)

for file in bucket_files:
    download_file(bucket_name, name=file, filename=file[1:])
