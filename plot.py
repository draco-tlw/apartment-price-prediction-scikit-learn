import matplotlib.pyplot as plt
import numpy as np


def plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_predict: np.ndarray,
    label: str,
    files_postfix: str,
):
    # --- price vs area
    plt.figure(figsize=(40, 25))
    plt.scatter(
        x_train[:, 0],
        y_train,
    )
    plt.scatter(
        x_train[:, 0],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Area ({label})", fontsize=24)
    plt.xlabel("Area", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(f"output_plots/model_on_price_vs_area_{files_postfix}_plot.png")

    # --- price vs year
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 1], y_train)
    plt.scatter(
        x_train[:, 1],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Year of built ({label})", fontsize=24)
    plt.xlabel("Year of built", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(f"output_plots/model_on_price_vs_year_{files_postfix}_plot.png")

    # --- price vs rooms
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 2], y_train)
    plt.scatter(
        x_train[:, 2],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Number of Rooms ({label})", fontsize=24)
    plt.xlabel("Number of Rooms", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(f"output_plots/model_on_price_vs_rooms_{files_postfix}_plot.png")

    # --- price vs total floors
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 3], y_train)
    plt.scatter(
        x_train[:, 3],
        y_predict,
    )
    plt.title(
        f"Model on Property Price vs. Building total floors ({label})", fontsize=24
    )
    plt.xlabel("Building total floors", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(
        f"output_plots/model_on_price_vs_building_total_floors_{files_postfix}_plot.png"
    )

    # --- price vs unit floor
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 4], y_train)
    plt.scatter(
        x_train[:, 4],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Unit floor ({label})", fontsize=24)
    plt.xlabel("Unit floor", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(f"output_plots/model_on_price_vs_unit_floor_{files_postfix}_plot.png")

    # --- price vs has elevator
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 5], y_train)
    plt.scatter(
        x_train[:, 5],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Has Elevetor ({label})", fontsize=24)
    plt.xlabel("Has Elevetor", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(f"output_plots/model_on_price_vs_has_elevator_{files_postfix}_plot.png")

    # --- price vs has parking
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 6], y_train)
    plt.scatter(
        x_train[:, 6],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Has Parking ({label})", fontsize=24)
    plt.xlabel("Has Parking", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(f"output_plots/model_on_price_vs_has_parking_{files_postfix}_plot.png")

    # --- price vs has storage unit
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 7], y_train)
    plt.scatter(
        x_train[:, 7],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Has Storage Unit ({label})", fontsize=24)
    plt.xlabel("Has Storage Unit", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(
        f"output_plots/model_on_price_vs_has_storage_unit_{files_postfix}_plot.png"
    )

    # --- price vs neighborhood mean
    plt.figure(figsize=(40, 25))
    plt.scatter(x_train[:, 8], y_train)
    plt.scatter(
        x_train[:, 8],
        y_predict,
    )
    plt.title(f"Model on Property Price vs. Neighborhood Mean ({label})", fontsize=24)
    plt.xlabel("Neighborhood Mean (Million Tomans)", fontsize=18)
    plt.ylabel("Price (Million Tomans)", fontsize=18)
    plt.grid(True)
    plt.ticklabel_format(style="plain", axis="both")
    plt.savefig(
        f"output_plots/model_on_price_vs_neighborhood_mean_{files_postfix}_plot.png"
    )
