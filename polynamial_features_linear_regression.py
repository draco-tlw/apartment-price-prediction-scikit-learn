import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from calculate_costs import calculate_costs
from plot import plot
from preprocess import preprocess

x_train, y_train = preprocess()

model = Pipeline([("poly", PolynomialFeatures(5)), ("linear", LinearRegression())])
model.fit(x_train, y_train)

model_name = "polynamial_features_linear_regression_d5"
joblib.dump(model, f"{model_name}.joblib")
print(f"Model saved successfully to '{model_name}.joblib'")

y_predict = model.predict(x_train)

plot(
    x_train,
    y_train,
    y_predict,
    model_name,
    model_name,
)
calculate_costs(y_train, y_predict)
