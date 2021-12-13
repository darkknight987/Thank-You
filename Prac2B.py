import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt

boston = load_boston()
boston_df = pd.DataFrame(data = boston.data, columns = boston.feature_names)
boston_df['MEDV'] = boston.target

print(boston_df)

corr_mat = boston_df.corr().round(2)
sns.set(rc = {'figure.figsize': (11.7, 8.27)})
sns.heatmap(data = corr_mat, annot = True)
plt.show()

