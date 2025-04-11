import os
import pandas as pd

PATH_TO_ALL_DATASETS = ''

SAVE_PATH = os.path.dirname(__file__)
DATASET = ['mnist', 'cifar10', 'food101', 'caltech256', 'stfd_dogs'][4]

DATASET_PATH = os.path.join(PATH_TO_ALL_DATASETS, DATASET)

df = pd.read_parquet(os.path.join(DATASET_PATH, 'annotation.parquet'))
df['path'] = df['path'].apply(lambda x: os.path.join(DATASET_PATH, x))

df.to_parquet(os.path.join(SAVE_PATH, f'{DATASET}_annotation.parquet'), index=False)
