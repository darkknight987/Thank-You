import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_boston

boston = load_boston()
feature_vals = pd.DataFrame(data = boston.data, columns = boston.feature_names)
target_vals = pd.DataFrame(data = boston.target)
y = np.array(target_vals).flatten()
n = len(y)
corr_vals = []
print(feature_vals)
print(target_vals)

for feature in boston.feature_names:
    x = feature_vals[feature]
    numerator = n * (sum(x * y)) - sum(x) * sum(y)
    d1 = n * sum(x * x) - sum(x) ** 2
    d2 = n * sum(y * y) - sum(y) ** 2
    r = numerator / math.sqrt(d1 * d2)
    corr_vals.append(r.round(2))

print("\nCorrelation values: \n", corr_vals)

def get_intercept_and_coeff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = sum((x - x_mean) * (y - y_mean))
    b1 = numerator / sum((x - x_mean) ** 2)
    b0 = y_mean - b1 * x_mean
    return (b0.round(3), round(b1, 3))

b0, b1 = get_intercept_and_coeff(feature_vals['RM'], y)
print("\nFor RM feature: ")
print("b1 = ", b1)
print("b0 = ", b0)

b0, b1 = get_intercept_and_coeff(feature_vals['LSTAT'], y)
print("\nFor LSTAT feature: ")
print("b1 = ", b1)
print("b0 = ", b0)


