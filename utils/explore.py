import pandas as pd

df_iter = pd.read_hdf("train.h5", key="table", chunksize=5)

for df in df_iter:
    print(df.head())
    print(df.shape)
    print(df.columns)
    break