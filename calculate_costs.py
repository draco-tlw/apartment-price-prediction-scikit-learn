import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_costs(y_train: np.ndarray, y_predict: np.ndarray):
    mae = mean_absolute_error(y_train, y_predict)

    mse = mean_squared_error(y_train, y_predict)
    rmse = np.sqrt(mse)

    return mae, rmse
