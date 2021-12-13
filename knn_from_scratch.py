import numpy as np
import pandas as pd
from collections import Counter
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                        columns = iris['feature_names'] + ['target'])

x = iris_df.iloc[:, : - 1]
y = iris_df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 0)

def get_predicted(x_train, y_train, x_test_pt, K):
    distances, y_train_lst = [], y_train.tolist()
    for row in range(len(x_train)):
        curr_train_pt = x_train.iloc[row, :].values.tolist()
        dist = 0
        for col in range(len(curr_train_pt)):
            dist += (curr_train_pt[col] - x_test_pt[col]) ** 2
        dist = np.sqrt(dist)
        distances.append(round(dist, 3))
    nearest_indices = np.array(distances).argsort()[ : K]
    y_nearest_labels = [y_train_lst[i] for i in nearest_indices]
    return Counter(y_nearest_labels).most_common()[0][0]

K = 5
y_preds = []
for i in range(len(x_test)):
    x_test_pt = x_test.iloc[i, :].values.tolist()
    y_preds.append(get_predicted(x_train, y_train, x_test_pt, K))

 
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_preds))
print("\nAccuracy: ", accuracy_score(y_test, y_preds))

print()

