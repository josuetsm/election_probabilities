import pandas as pd
from datetime import datetime

df = pd.read_csv('data/docs/dt2_info.csv')

df['atime'] = df['atime'].apply(lambda x: datetime.fromtimestamp(x))
df['mtime'] = df['mtime'].apply(lambda x: datetime.fromtimestamp(x))