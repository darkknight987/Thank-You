import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def perform_poly_reg(df, feature, target):
    rmse_train, rmse_test, models, highest_deg = [], [], [], 4
    df = df.sort_values(by = [feature])
    x, y = df[feature], df[target]
    
    x_train_reshaped = np.array(x_train).reshape(-1, 1)
    x_test_reshaped = np.array(x_test).reshape(-1, 1)
    degrees = range(1, highest_deg + 1)
    for deg in degrees:
        poly = PolynomialFeatures(degree = deg)
        model = LinearRegression()
        x_train_poly = poly.fit_transform(x_train_reshaped)
        model.fit(x_train_poly, y_train)
        y_train_poly_pred = model.predict(x_train_poly)
        x_test_poly = poly.fit_transform(x_test_reshaped)
        y_test_poly_pred = model.predict(x_test_poly)
        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_poly_pred)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_poly_pred)))
        models.append(model)

    # plotting rmse values
    x_axis = np.arange(len(rmse_train))
    plt.plot(degrees, rmse_train, label = "RMSE train", marker = 'o')
    plt.plot(degrees, rmse_test, label = "RMSE test", marker = 'o')
    plt.title("RMSE values")
    plt.xlabel("Degrees")
    plt.ylabel("Error values")
    plt.legend()
    plt.show()

    index_min = np.argmin(list(map(lambda x, y: x + y, rmse_train, rmse_test)))
    optimal_model = models[index_min]
    x = np.array(x).reshape(-1, 1)
    poly = PolynomialFeatures(degree = index_min + 1)
    x_poly = poly.fit_transform(x)
    y_pred = optimal_model.predict(x_poly)

    # plot optimal ploynomial curve
    plt.title("Optimal polynomial curve with deg = " + str(index_min + 1))
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.scatter(x, y, s = 15)
    plt.plot(x, y_pred, color = 'k')
    plt.show()

    print("RMSE for training: ", rmse_train[index_min])
    print("RMSE for testing: ", rmse_test[index_min])
    print("Optimal degree = ", index_min + 1)


# boston dataset
boston = load_boston()
boston_df = pd.DataFrame(data = boston.data, columns = boston.feature_names)
boston_df['MEDV'] = boston.target

# diabetes dataset
diab_df = pd.read_csv(r"D:\Msc CS Sem 1\Practicals\Machine Learning\Sample Data\diabetes.csv")

print("\nFor boston housing dataset: ")
perform_poly_reg(boston_df, feature = 'LSTAT', target = 'MEDV')
print("\nFor diabetes dataset: ")
perform_poly_reg(diab_df, feature = 'bmi', target = 'y')
