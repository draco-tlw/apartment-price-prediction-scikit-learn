import joblib
from sklearn.linear_model import LinearRegression

from calculate_costs import calculate_costs
from plot import plot
from preprocess import preprocess

x_train, y_train = preprocess()

model = LinearRegression()
model.fit(x_train, y_train)

b = model.intercept_
w = model.coef_

print("model parameters:")
print(f"w: {w}")
print(f"b: {b}")

model_name = "linear_regression"
joblib.dump(model, f"{model_name}.joblib")
print(f"Model saved successfully to '{model_name}.joblib'")

y_predict = model.predict(x_train.reshape(-1, 9))

plot(
    x_train,
    y_train,
    y_predict,
    model_name,
    model_name,
)
calculate_costs(y_train, y_predict)
