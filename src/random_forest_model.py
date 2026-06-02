# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# %%
def train_random_forest(df, target_column, crop_name):

    features = [
        col for col in df.columns
        if col not in [
            "year",
            target_column
        ]
    ]

    model_df = df[
        features + [target_column, "year"]
    ].copy()

    model_df = model_df.dropna()

    if len(model_df) < 8:
        print(f"\nNot enough data for {crop_name}. Skipping.")
        return None, None, None

    X = model_df[features]
    y = model_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame({
        "year": model_df.loc[y_test.index, "year"],
        "actual_price": y_test.values,
        "predicted_price": y_pred
    })

    metrics = {
        "crop": crop_name,
        "model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, results, metrics

# %%
def run_random_forest_models(df):

    crops = {
        "Wheat": "wheat_mean",
        "Maize": "maize_mean",
        "Barley": "barley_mean",
        "Sorghum": "sorghum_mean",
        "Rice": "rice_mean"
    }

    all_results = {}
    metrics_list = []

    for crop_name, column in crops.items():

        if column not in df.columns:
            continue

        model, results, metrics = train_random_forest(
            df,
            column,
            crop_name
        )
        if model is None:
            continue

        print(f"\nRANDOM FOREST RESULTS FOR {crop_name.upper()}")

        print("MAE:", metrics["MAE"])
        print("RMSE:", metrics["RMSE"])
        print("R2:", metrics["R2"])

        all_results[crop_name] = results

        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    print("\nRANDOM FOREST MODEL COMPARISON:")
    print(metrics_df)

    return all_results, metrics_df

# %%
def plot_random_forest_results(all_results):

    import math

    n = len(all_results)

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 5 * rows)
    )

    axes = axes.flatten()

    for ax, (crop_name, results) in zip(
        axes,
        all_results.items()
    ):

        ax.plot(
            results["year"],
            results["actual_price"],
            marker="o",
            label="Actual"
        )

        ax.plot(
            results["year"],
            results["predicted_price"],
            marker="o",
            linestyle="--",
            label="Random Forest"
        )

        ax.set_title(
            f"{crop_name} Random Forest Forecast"
        )

        ax.grid(True)

        ax.legend()

    for i in range(
        len(all_results),
        len(axes)
    ):
        fig.delaxes(axes[i])

    plt.tight_layout()
    from src.save_plots import save_current_figure
    save_current_figure(
        "random_forest_forecasts.png"
    )
    plt.show()

