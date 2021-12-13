import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"D:\Jayesh\Msc CS Sem 1\Practicals\Machine Learning\Sample Data\User_Data.csv")
df["Gender"].replace({"Male": 0, "Female": 1}, inplace=True)
x = df.iloc[:, 1: 4].values
y = df.iloc[:, 4].values

scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), Normalizer()]

def perform_logistic_regr(x, y, scaler):
    scaled_x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size = 0.25, random_state = 0)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy score: ", accuracy_score(y_test, y_pred))

for scaler in scalers:
    print("\nUsing {}:".format(scaler.__class__.__name__))
    perform_logistic_regr(x, y, scaler)

print()
