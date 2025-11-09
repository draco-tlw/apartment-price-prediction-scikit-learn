from typing import cast

import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from calculate_costs import calculate_costs
from plot import plot
from preprocess import preprocess

x_train, y_train = preprocess()
x_train, x_cv, y_train, y_cv = cast(
    list[np.ndarray], train_test_split(x_train, y_train, test_size=0.2, random_state=42)
)

model_name = "sgd_regressor"
scaler = StandardScaler()

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("sgdr", SGDRegressor(max_iter=100000)),
    ]
)
model.fit(x_train, y_train)

y_predict: np.ndarray = model.predict(x_train)
mae, rmse = calculate_costs(y_train, y_predict)
print(f"MAE train: {mae:.2f}, RMSE train: {rmse:.2f}")

y_predict: np.ndarray = model.predict(x_cv)
mae, rmse = calculate_costs(y_cv, y_predict)
print(f"MAE cross-validation: {mae:.2f}, RMSE cross-validation: {rmse:.2f}")

plot(
    x_cv,
    y_cv,
    y_predict,
    model_name,
    model_name,
)

print("errors: ", x_cv[y_predict < 500])


model.fit(
    np.concatenate((x_train, x_cv), axis=0), np.concatenate((y_train, y_cv), axis=0)
)
joblib.dump(model, f"{model_name}.joblib")
print(f"Model saved successfully to '{model_name}.joblib'")
