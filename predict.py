import joblib
import numpy as np
import pandas as pd

neighborhoods_df = pd.read_csv("./neighborhoods_table.csv", index_col="neighborhood")
neighborhood_price_means = neighborhoods_df["neighborhood_price_mean"]

model = joblib.load("./polynamial_features_linear_regression_d2.joblib")

data_point = np.array(
    [
        90,  # area_sqm
        1395,  # year_built
        2,  # num_rooms
        5,  # building_total_floors
        3,  # unit_floor
        1,  # has_elevator
        1,  # has_parking
        1,  # has_storage_unit
        neighborhood_price_means.loc["گلشور"],  # neighborhood_price_mean
    ]
)

x = data_point.reshape(1, -1)

y = model.predict(x)

print(f"Input features: {data_point.tolist()}")
print("---")
print(f"Predicted price: {y[0]:.2f} million Tomans")
