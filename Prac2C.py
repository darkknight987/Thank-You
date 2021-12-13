import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def perform_lin_reg(x, y, plot_graph = False, labels = ('', '')):
    tot_var = 1 if len(x.shape) == 1 else x.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)
    x_train_reshaped = np.array(x_train).reshape(-1, tot_var)
    x_test_reshaped = np.array(x_test).reshape(-1, tot_var)
    
    lin_model = LinearRegression()
    lin_model.fit(x_train_reshaped, y_train)
    y_train_pred = lin_model.predict(x_train_reshaped)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    y_test_pred = lin_model.predict(x_test_reshaped)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("Intercept: ", lin_model.intercept_)
    print("Coefficient: ",lin_model.coef_)
    print("Model's performance for training set - RMSE: ", rmse_train)
    print("Model's performance for testing set - RMSE: ", rmse_test)

    if plot_graph:
        plt.scatter(x, y, color="#268abf",marker = "o", s = 10)
        pred = lin_model.predict(x_test_reshaped)
        plt.plot(x_test_reshaped, pred, color = "#3c4a52")
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.show()

def caller(df, feature1, feature2, target):
    x1 = df[feature1]
    x2 = df[feature2]
    x3 = pd.DataFrame(np.c_[df[feature1], df[feature2]], columns = [feature1, feature2])
    y = df[target]
    print("\nFor feature '{}' and target '{}'".format(feature1, target))
    perform_lin_reg(x1, y, plot_graph = True, labels = (feature1, target))
    print("\nFor feature '{}' and target '{}'".format(feature2, target))
    perform_lin_reg(x2, y, plot_graph = True, labels = (feature2, target))
    print("\nFor feature '{}', '{}' and target '{}'".format(feature1, feature2, target))
    perform_lin_reg(x3, y)

# boston dataset
boston = load_boston()
boston_df = pd.DataFrame(data = boston.data, columns = boston.feature_names)
boston_df['MEDV'] = boston.target

# diabetes dataset
diab_df = pd.read_csv(r"D:\Jayesh\Msc CS Sem 1\Practicals\Machine Learning\Sample Data\diabetes.csv")

caller(boston_df, 'LSTAT', 'RM', 'MEDV')
caller(diab_df, 'bmi', 'ltg', 'y')
