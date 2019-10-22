import pandas as pd
import numpy as np
df=pd.read_csv("weights.csv",header=None)
labels=pd.read_csv("labels.csv",header=None)
print(np.array(df.values)[0])
print(df.values.shape[1])