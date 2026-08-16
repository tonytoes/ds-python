# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# %%
