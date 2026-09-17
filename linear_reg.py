import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_regression():

    # print(pd.read_csv("Walmart_Sales.csv").info())

    col_name = ["Holiday_Flag","Temperature","Fuel_Price","CPI"]
    X = pd.DataFrame()

    for i in col_name:
        X["w0"] = 1
        X[i] = pd.read_csv("Walmart_Sales.csv")[i]

    # print(X) #X is a (6435 x 5) matrix with intercept column and turn into numpy array for linear algebra
    X_matrix= X.to_numpy()

    Y = pd.DataFrame()
    Y["Weekly_Sales"] = pd.read_csv("Walmart_Sales.csv")["Weekly_Sales"]
    # Y is a (6435 x 1) matrix and turn into numpy array for linear algebra
    Y_matrix = Y.to_numpy()

    # finding XT (X transpose)

    X_t = X_matrix.T

    # #Find XtX

    XtX = X_t @ X
    # print(XtX, "XTX")

    XtY = X_t@Y_matrix

    #Use Rank to check matrix XTX invertibility --> using rank using detmerminant in python can be very small and floating points can be consireded as 0
    if np.linalg.matrix_rank(XtX) == XtX.shape[0]:
        w_star = np.linalg.inv(XtX)@XtY
        print(w_star.shape)

    return w_star


def test_reg(w_star):
    col_name = ["Holiday_Flag","Temperature","Fuel_Price","CPI"]
    X = pd.DataFrame()

    for i in col_name:
        X["w0"] = 1
        X[i] = pd.read_csv("test_walmart_sales.csv")[i]
    
    X_matrix = X.to_numpy()
    print(X_matrix.shape)

    y_pred = X_matrix @ w_star
    y_actual = pd.read_csv("test_walmart_sales.csv")["Weekly_Sales"]
    mse = np.mean((y_actual.to_numpy().flatten() - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    # Convert y_actual from DataFrame to 1D array
    # y_actual_array = y_actual.to_numpy().flatten()

    # plt.figure(figsize=(7, 5))
    # plt.scatter(y_actual_array, y_pred)

    # # Reference line y = x
    # minimum = min(y_actual_array.min(), y_pred.min())
    # maximum = max(y_actual_array.max(), y_pred.max())

    # plt.plot([minimum, maximum], [minimum, maximum])

    # plt.xlabel("Actual Weekly Sales")
    # plt.ylabel("Predicted Weekly Sales")
    # plt.title("Actual vs Predicted Weekly Sales")
    # plt.show()

    return y_pred

w_s = train_regression()
test_reg(w_s)
