import numpy as np
import pandas as pd


def preprocess():
    dataset = pd.read_csv("./dataset-2025-10-18_17-04-01.csv")
    pruned_dataset: pd.DataFrame = dataset.loc[
        (dataset["area_sqm"] <= 400)
        & (dataset["area_sqm"] >= 50)
        & (dataset["price_tomans"] <= 40_000_000_000)
        & (dataset["price_tomans"] >= 1_000_000_000)
        & (dataset["year_built"] >= 1370)
        & (dataset["building_total_floors"] <= 15)
        & (dataset["building_total_floors"] > 1)
    ].copy()

    pruned_dataset["price_mtomans"] = pruned_dataset["price_tomans"] / 1_000_000

    neighborhood_price_means = pruned_dataset.groupby("neighborhood")[
        "price_mtomans"
    ].mean()

    neighborhood_counts = pruned_dataset.groupby("neighborhood")["neighborhood"].count()

    pruned_dataset["neighborhood_price_mean"] = pruned_dataset["neighborhood"].map(
        lambda n: neighborhood_price_means.loc[n]
    )

    pruned_dataset["neighborhood_apartment_count"] = pruned_dataset["neighborhood"].map(
        lambda n: neighborhood_counts.loc[n]
    )

    pruned_dataset = pruned_dataset.loc[
        pruned_dataset["neighborhood_apartment_count"] >= 5
    ].copy()

    neighborhoods_df: pd.DataFrame = pruned_dataset.loc[
        :, ["neighborhood", "neighborhood_price_mean", "neighborhood_apartment_count"]
    ].drop_duplicates()

    neighborhoods_df.to_csv("neighborhoods_table.csv", index=False)

    print(f"{dataset.shape[0] - pruned_dataset.shape[0]} items pruned")

    x_features = [
        "area_sqm",
        "year_built",
        "num_rooms",
        "building_total_floors",
        "unit_floor",
        "has_elevator",
        "has_parking",
        "has_storage_unit",
        "neighborhood_price_mean",
    ]
    y_target = "price_mtomans"

    x_train: np.ndarray = pruned_dataset.loc[:, x_features].to_numpy()
    y_train: np.ndarray = pruned_dataset.loc[:, y_target].to_numpy()

    return x_train, y_train
