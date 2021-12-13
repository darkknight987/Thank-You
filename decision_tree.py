import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image  
from six import StringIO  
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_file = r"D:\Jayesh\Msc CS Sem 1\Practicals\Machine Learning\Sample Data\PastHires.csv"
df = pd.read_csv(input_file, header = 0)
print("\nData before:\n", df.head())

d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
print("\nData after:\n", df.head())

features = list(df.columns[:6])

y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 3, random_state = 10)
clf = clf.fit(X,y)

dot_data = StringIO()  
tree.export_graphviz(clf, out_file = dot_data, feature_names = features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("temp.png")
img = mpimg.imread('temp.png')
plt.imshow(img)
plt.show()

print()