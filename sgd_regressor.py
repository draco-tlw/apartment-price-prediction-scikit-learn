import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from calculate_costs import calculate_costs
from plot import plot
from preprocess import preprocess

x_train, y_train = preprocess()

scaler = StandardScaler()
x_train_norm: np.ndarray = scaler.fit_transform(x_train)

model = SGDRegressor(max_iter=100000)
model.fit(x_train_norm, y_train)

b = model.intercept_
w = model.coef_

print("model parameters:")
print(f"w: {w}")
print(f"b: {b}")

model_name = "sgd_regressor"
joblib.dump(model, f"{model_name}.joblib")
print(f"Model saved successfully to '{model_name}.joblib'")

y_predict = model.predict(x_train_norm.reshape(-1, 9))

plot(
    x_train,
    y_train,
    y_predict,
    model_name,
    model_name,
)
calculate_costs(y_train, y_predict)
