import numpy as np
import pandas as pd

annot_path = '/mnt/disk2/mironov/datasets/cifar10/annotation.parquet'

df = pd.read_parquet(annot_path)

# Assuming df is your original DataFrame and 'set' is the column indicating the dataset split
test_df = df[df['set'] == 'test']

# Randomly select 5000 indices from the test_df
val_indices = np.random.choice(test_df.index, size=5000, replace=False)

# Create the validation DataFrame using the selected indices
val_df = test_df.loc[val_indices]
val_df['set'] = 'valid'

# Create the new test DataFrame by excluding the rows in val_indices
test_df_n = test_df.loc[~test_df.index.isin(val_indices)]

new_df = pd.concat([df[df['set'] == 'train'], val_df, test_df_n], ignore_index=True)

new_df.to_parquet(annot_path)